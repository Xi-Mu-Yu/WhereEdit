"""Attention-Guided Automatic Local Edit Mask (AutoMask) for WhereEdit."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:  # pragma: no cover - optional dependency
    _scipy_ndimage = None

if TYPE_CHECKING:
    from pipeline_whereedit import WhereEditPipeline

LOGGER = logging.getLogger(__name__)


@dataclass
class _AttnCaptureSpec:
    batch_index: int
    token_indices: torch.Tensor
    storage: List[torch.Tensor] = field(default_factory=list)


class _CapturingAttnProcessor:
    """Cross-attention processor that captures edit-token maps in a single forward pass."""

    def __init__(
        self,
        inner_processor: Any,
        capture_specs: Optional[List[_AttnCaptureSpec]] = None,
    ) -> None:
        self.inner_processor = inner_processor
        self.capture_specs = list(capture_specs or [])
        self._cached_token_indices: Dict[int, torch.Tensor] = {}

    def set_capture_specs(self, capture_specs: List[_AttnCaptureSpec]) -> None:
        self.capture_specs = list(capture_specs)
        self._cached_token_indices.clear()

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        active_specs = [
            spec
            for spec in self.capture_specs
            if spec.token_indices.numel() > 0
        ]
        if encoder_hidden_states is None or not active_specs:
            return self.inner_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        try:
            return self._forward_cross_attn_with_capture(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                active_specs,
            )
        except Exception as exc:  # pragma: no cover - diffusers version drift
            LOGGER.debug(
                "Single-pass cross-attention capture failed in %s: %s",
                type(attn).__name__,
                exc,
            )
            return self.inner_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

    def _resolve_token_indices(
        self,
        spec: _AttnCaptureSpec,
        device: torch.device,
        seq_len: int,
    ) -> torch.Tensor:
        cached = self._cached_token_indices.get(spec.batch_index)
        if cached is not None and cached.device == device:
            token_idx = cached
        else:
            token_idx = spec.token_indices.to(device=device, dtype=torch.long)
            self._cached_token_indices[spec.batch_index] = token_idx
        return token_idx[(token_idx >= 0) & (token_idx < seq_len)]

    @staticmethod
    def _store_capture(
        probs: torch.Tensor,
        attn: Any,
        spec: _AttnCaptureSpec,
        token_idx: torch.Tensor,
    ) -> None:
        if token_idx.numel() == 0:
            return

        num_heads = getattr(attn, "heads", None) or getattr(attn, "num_heads", 1)
        head_slice = slice(spec.batch_index * num_heads, (spec.batch_index + 1) * num_heads)
        batch_probs = probs[head_slice]
        if batch_probs.shape[0] == 0:
            return

        token_attn = batch_probs[:, :, token_idx].mean(dim=-1)
        spatial = token_attn.shape[1]
        side = int(spatial**0.5)
        if side * side != spatial:
            return
        token_attn = token_attn.reshape(batch_probs.shape[0], side, side).mean(dim=0, keepdim=True)
        spec.storage.append(token_attn.detach())

    def _forward_cross_attn_with_capture(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        active_specs: List[_AttnCaptureSpec],
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if hasattr(attn, "get_attention_scores"):
            probs = attn.get_attention_scores(query, key, attention_mask)
        else:
            scale = getattr(attn, "scale", query.shape[-1] ** -0.5)
            scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = scores.softmax(dim=-1)

        seq_len = probs.shape[-1]
        for spec in active_specs:
            token_idx = self._resolve_token_indices(spec, probs.device, seq_len)
            self._store_capture(probs, attn, spec, token_idx)

        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _to_bchw_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize an attention / mask tensor to (N, C, H, W)."""
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected 2D-4D mask, got shape {tuple(tensor.shape)}")


class AutoMask:
    """
    Attention-guided automatic local edit mask.

    Aggregates cross-attention responses from inserted target tokens and removed
    source tokens to localize editable regions without external annotations.
    """

    def __init__(self, pipeline: WhereEditPipeline) -> None:
        self._pipeline = pipeline
        self._attn2_modules: Dict[str, Any] = {}
        self._attn_original_processors: Dict[str, Any] = {}
        self._attn_capture_processors: Dict[str, _CapturingAttnProcessor] = {}
        self._gaussian_kernel_cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._special_token_ids = self._build_special_token_ids()
        for name, module in pipeline.unet.named_modules():
            if "attn2" in name and hasattr(module, "get_processor"):
                self._attn2_modules[name] = module

    @property
    def tokenizer(self):
        return self._pipeline.tokenizer

    @property
    def unet(self):
        return self._pipeline.unet

    def prompt_token_indices(
        self,
        source_prompt: str,
        target_prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (edit, removed, content) token indices in a single tokenization pass."""
        src_ids = self.tokenizer(
            source_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        tgt_ids = self.tokenizer(
            target_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        src_list = src_ids.tolist()
        tgt_list = tgt_ids.tolist()
        content_flags = torch.zeros(len(tgt_list), dtype=torch.bool)
        removed_flags = torch.zeros(len(src_list), dtype=torch.bool)
        edit_flags = torch.zeros(len(tgt_list), dtype=torch.bool)
        for op, a0, a1, b0, b1 in SequenceMatcher(None, src_list, tgt_list).get_opcodes():
            if op == "equal":
                end = min(b1, len(tgt_list))
                for idx in range(b0, end):
                    if self._is_valid_token(tgt_list[idx]):
                        content_flags[idx] = True
            elif op in ("replace", "delete"):
                for idx in range(a0, min(a1, len(src_list))):
                    if self._is_valid_token(src_list[idx]):
                        removed_flags[idx] = True
            if b0 < len(tgt_list) and op in ("replace", "insert"):
                end = min(b1, len(tgt_list))
                for idx in range(b0, end):
                    if self._is_valid_token(tgt_list[idx]):
                        edit_flags[idx] = True

        edit_indices = torch.nonzero(edit_flags, as_tuple=False).view(-1)
        if edit_indices.numel() == 0:
            edit_indices = torch.tensor(
                [idx for idx, tok in enumerate(tgt_list) if self._is_valid_token(tok)],
                dtype=torch.long,
            )
        return (
            edit_indices,
            torch.nonzero(removed_flags, as_tuple=False).view(-1),
            torch.nonzero(content_flags, as_tuple=False).view(-1),
        )

    def compute_subject_mask(
        self,
        x_anchor: torch.Tensor,
        src_embed: torch.Tensor,
        edit_embed: torch.Tensor,
        noise: torch.Tensor,
        t_s: float,
        token_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate cross-attention for edit/removed tokens; content tokens are a fallback."""
        batch = x_anchor.shape[0]
        device = x_anchor.device
        zero = torch.zeros(
            batch,
            1,
            x_anchor.shape[-2],
            x_anchor.shape[-1],
            device=device,
            dtype=x_anchor.dtype,
        )

        edit_indices, removed_indices, content_indices = token_indices
        edit_attn, removed_attn = self._capture_attn_maps_batched(
            x_anchor,
            [
                (edit_embed, edit_indices),
                (src_embed, removed_indices),
            ],
            noise,
            t_s,
        )[:2]

        attn_subject: Optional[torch.Tensor] = None
        if self._attn_has_signal(removed_attn) and self._attn_has_signal(edit_attn):
            attn_subject = torch.maximum(removed_attn, edit_attn)
        elif self._attn_has_signal(removed_attn):
            attn_subject = removed_attn
        elif self._attn_has_signal(edit_attn):
            attn_subject = edit_attn
        else:
            content_results = self._capture_attn_maps_batched(
                x_anchor,
                [(edit_embed, content_indices)],
                noise,
                t_s,
            )
            content_attn = content_results[0]
            if self._attn_has_signal(content_attn):
                attn_subject = content_attn

        if attn_subject is None:
            return zero

        peak = attn_subject.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (attn_subject / peak).clamp(0.0, 1.0).to(dtype=x_anchor.dtype)

    def refine_subject_mask(
        self,
        attn_subject_mask: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Spatial edit region from cross-attention on changed prompt tokens."""
        attn_norm = self._normalize_mask(attn_subject_mask)


        threshold_ratio = 0.5
        return self._peak_confidence_mask(
            attn_norm,
            edge_buffer=int(params["edit_mask_edge_buffer"]),
            feather_kernel=int(params["edit_mask_feather"]),
            feather_passes=int(params["edit_mask_feather_passes"]),
            threshold_ratio=threshold_ratio,
            support_threshold=0.,
        )

    @staticmethod
    def apply_local_latent_edit(
        x_src: torch.Tensor,
        x_curr: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pin background latents to the source image; keep edits only inside the mask."""
        return x_src * (1.0 - edit_mask) + x_curr * edit_mask

    def _build_special_token_ids(self) -> frozenset[int]:
        skip = {
            tok
            for tok in (
                getattr(self.tokenizer, "pad_token_id", None),
                getattr(self.tokenizer, "bos_token_id", None),
                getattr(self.tokenizer, "eos_token_id", None),
                0,
            )
            if tok is not None
        }
        return frozenset(int(tok) for tok in skip)

    def _is_valid_token(self, token_id: int) -> bool:
        return int(token_id) not in self._special_token_ids

    def _install_attn_capture(self, capture_specs: List[_AttnCaptureSpec]) -> None:
        for name, module in self._attn2_modules.items():
            if name not in self._attn_original_processors:
                self._attn_original_processors[name] = module.get_processor()
            if name not in self._attn_capture_processors:
                self._attn_capture_processors[name] = _CapturingAttnProcessor(
                    self._attn_original_processors[name],
                )
            processor = self._attn_capture_processors[name]
            processor.set_capture_specs(capture_specs)
            module.set_processor(processor)

    def _restore_attn_processors(self) -> None:
        for name, module in self._attn2_modules.items():
            original = self._attn_original_processors.get(name)
            if original is not None:
                module.set_processor(original)

    def _aggregate_attn_storage(
        self,
        storage: List[torch.Tensor],
        *,
        batch: int,
        target_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not storage:
            return None

        by_size: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        for attn_map in storage:
            map_4d = _to_bchw_mask(attn_map).float()
            by_size.setdefault((map_4d.shape[-2], map_4d.shape[-1]), []).append(map_4d)

        resized_groups: List[torch.Tensor] = []
        for spatial, group in by_size.items():
            stacked = torch.cat(group, dim=0)
            if spatial != target_size:
                stacked = torch.nn.functional.interpolate(
                    stacked,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_groups.append(stacked)

        attn_subject = torch.cat(resized_groups, dim=0).mean(dim=0, keepdim=True)
        if attn_subject.shape[0] == 1 and batch > 1:
            attn_subject = attn_subject.expand(batch, -1, -1, -1)
        peak = attn_subject.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (attn_subject / peak).clamp(0.0, 1.0).to(device=device, dtype=dtype)

    def _capture_attn_maps_batched(
        self,
        x_anchor: torch.Tensor,
        captures: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        noise: torch.Tensor,
        t_s: float,
    ) -> List[Optional[torch.Tensor]]:
        active: List[Tuple[torch.Tensor, torch.Tensor, _AttnCaptureSpec]] = []
        for cond_embed, token_indices in captures:
            if token_indices.numel() == 0:
                continue
            active.append(
                (
                    cond_embed,
                    token_indices,
                    _AttnCaptureSpec(batch_index=len(active), token_indices=token_indices),
                )
            )

        results: List[Optional[torch.Tensor]] = [None] * len(captures)
        if not active:
            return results

        batch = x_anchor.shape[0]
        device = x_anchor.device
        capture_specs = [spec for _, _, spec in active]
        self._install_attn_capture(capture_specs)
        try:
            t_idx = self._pipeline._time_to_index(batch, t_s, device=device)
            alpha_t, sigma_t = self._pipeline._get_alpha_sigma(x_anchor, t_idx)
            z_t = alpha_t * x_anchor + sigma_t * noise

            cond_embed = torch.cat([cond for cond, _, _ in active], dim=0)
            z_batch = z_t.repeat(len(active), 1, 1, 1)
            t_batch = t_idx.repeat(len(active))

            self.unet(
                sample=z_batch,
                timestep=t_batch,
                encoder_hidden_states=cond_embed,
                return_dict=False,
            )
        finally:
            self._restore_attn_processors()

        target_size = (x_anchor.shape[-2], x_anchor.shape[-1])
        active_idx = 0
        for capture_idx, (_, token_indices) in enumerate(captures):
            if token_indices.numel() == 0:
                continue
            _, _, spec = active[active_idx]
            results[capture_idx] = self._aggregate_attn_storage(
                spec.storage,
                batch=batch,
                target_size=target_size,
                device=device,
                dtype=x_anchor.dtype,
            )
            active_idx += 1
        return results

    @staticmethod
    def _attn_has_signal(attn_map: Optional[torch.Tensor]) -> bool:
        return attn_map is not None and bool(attn_map.max().item() > 0.0)

    @staticmethod
    def _connected_component_with_seed(binary: np.ndarray, seed_y: int, seed_x: int) -> np.ndarray:
        height, width = binary.shape
        if not binary[seed_y, seed_x]:
            component = np.zeros((height, width), dtype=bool)
            component[seed_y, seed_x] = True
            return component

        if _scipy_ndimage is not None:
            labeled, _ = _scipy_ndimage.label(binary)
            seed_label = labeled[seed_y, seed_x]
            if seed_label == 0:
                component = np.zeros((height, width), dtype=bool)
                component[seed_y, seed_x] = True
                return component
            return labeled == seed_label

        visited = np.zeros((height, width), dtype=bool)
        stack = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and not visited[ny, nx]
                    and binary[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        return visited

    def _get_gaussian_kernels(
        self,
        mask: torch.Tensor,
        kernel_size: int,
        sigma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = mask.shape[1]
        cache_key = (mask.device, mask.dtype, kernel_size, channels, round(sigma, 4))
        cached = self._gaussian_kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        coords = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - kernel_size // 2
        gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel_h = gauss.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
        kernel_v = gauss.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
        self._gaussian_kernel_cache[cache_key] = (kernel_h, kernel_v)
        return kernel_h, kernel_v

    @staticmethod
    def _gaussian_blur_mask(
        mask: torch.Tensor,
        kernel_size: int,
        sigma: Optional[float] = None,
        kernel_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        if kernel_size <= 1:
            return mask
        if kernel_size % 2 == 0:
            kernel_size += 1
        if sigma is None:
            sigma = max(1.0, kernel_size / 3.0)
        if kernel_cache is not None:
            kernel_h, kernel_v = kernel_cache._get_gaussian_kernels(mask, kernel_size, sigma)
        else:
            coords = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - kernel_size // 2
            gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
            gauss = gauss / gauss.sum()
            channels = mask.shape[1]
            kernel_h = gauss.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
            kernel_v = gauss.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(mask, kernel_h, padding=(0, padding), groups=mask.shape[1])
        blurred = torch.nn.functional.conv2d(blurred, kernel_v, padding=(padding, 0), groups=mask.shape[1])
        return blurred

    def _feather_subject_mask(
        self,
        mask: torch.Tensor,
        *,
        edge_buffer: int,
        feather_kernel: int,
        feather_passes: int,
        support_threshold: float = 0.1,
    ) -> torch.Tensor:
        support = (mask > support_threshold).float()
        expanded = support
        if edge_buffer > 0:
            dilate_kernel = edge_buffer * 2 + 1
            expanded = torch.nn.functional.max_pool2d(
                support,
                kernel_size=dilate_kernel,
                stride=1,
                padding=edge_buffer,
            )
        refined = expanded * mask.clamp(0.0, 1.0)
        for _ in range(feather_passes):
            refined = self._gaussian_blur_mask(refined, feather_kernel, kernel_cache=self)
        refined = refined * expanded
        peak = refined.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (refined / peak).clamp(0.0, 1.0)

    def _peak_confidence_mask(
        self,
        mask: torch.Tensor,
        *,
        edge_buffer: int,
        feather_kernel: int,
        feather_passes: int,
        threshold_ratio: float = 0.5,
        support_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Select the peak-confidence region, expand slightly, then feather edges."""
        mask_cpu = mask.detach().float().cpu()
        cores: List[torch.Tensor] = []
        for batch_idx in range(mask_cpu.shape[0]):
            values = mask_cpu[batch_idx, 0]
            peak_flat = int(values.argmax())
            peak_y, peak_x = np.unravel_index(peak_flat, values.shape)
           
            threshold = threshold_ratio 
            binary = values.numpy() >= threshold
            component = self._connected_component_with_seed(binary, peak_y, peak_x)
            cores.append(torch.from_numpy(component.astype(np.float32)) * values)

        subject = torch.stack(cores, dim=0).unsqueeze(1).to(device=mask.device, dtype=mask.dtype)
        return self._feather_subject_mask(
            subject,
            edge_buffer=edge_buffer,
            feather_kernel=feather_kernel,
            feather_passes=feather_passes,
            support_threshold=support_threshold,
        )

    @staticmethod
    def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
        peak = mask.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (mask / peak).clamp(0.0, 1.0)

    