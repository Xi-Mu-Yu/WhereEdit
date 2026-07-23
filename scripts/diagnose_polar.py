#!/usr/bin/env python3
"""Compare linear transport vs ACT on one PIE sample (u_hat statistics)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from act import ACT  # noqa: E402
from pipeline_whereedit import WhereEditPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=ROOT / "sd-turbo")
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "sd-turbo/pie_bench/annotation_images/0_random_140/000000000000.jpg",
    )
    parser.add_argument("--source-prompt", default="a slanted mountain bicycle on the road in front of a building")
    parser.add_argument(
        "--target-prompt",
        default="a slanted [rusty] mountain bicycle on the road in front of a building",
    )
    parser.add_argument("--t-start", type=float, default=0.9)
    parser.add_argument("--t-delta", type=float, default=0.15)
    args = parser.parse_args()

    paths = {
        "unet_path": str(args.model_root / "unet"),
        "scheduler_path": str(args.model_root / "scheduler"),
        "vae_path": str(args.model_root / "vae"),
        "tokenizer_path": str(args.model_root / "tokenizer"),
        "text_encoder_path": str(args.model_root / "text_encoder"),
    }
    pipe = WhereEditPipeline.from_local_weights(paths, device="cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image).convert("RGB")
    pixel = pipe._prepare_image_tensor(image)
    latents = pipe._encode_image_to_latent(pixel)
    src = pipe.encode_prompt([args.source_prompt])
    tgt = pipe.encode_prompt([args.target_prompt])
    noise = pipe._prepare_noise_list(latents, 42, 1)

    batch, device = latents.shape[0], latents.device
    t_s, delta = args.t_start, args.t_delta
    t_idx_s = pipe._time_to_index(batch, t_s, device=device)
    t_idx_s0 = pipe._time_to_index(batch, max(0.0, t_s - delta), device=device)
    alpha_s, sigma_s = pipe._get_alpha_sigma(latents, t_idx_s)
    alpha_prev, sigma_prev = pipe._get_alpha_sigma(latents, t_idx_s0)
    z_s = alpha_s * latents + sigma_s * noise[0]
    z_prev = alpha_prev * latents + sigma_prev * noise[0]

    def pred_x0(z, t_idx, cond):
        tp = torch.full((batch,), t_idx[0].item(), device=device, dtype=torch.long)
        alpha, sigma = pipe._get_alpha_sigma(z, tp)
        eps = pipe.unet(sample=z, timestep=tp, encoder_hidden_states=cond, return_dict=False)[0]
        return (z - sigma * eps) / alpha

    dv_s = pred_x0(z_s, t_idx_s, tgt) - pred_x0(z_s, t_idx_s, src)
    dv_s0 = pred_x0(z_prev, t_idx_s0, tgt) - pred_x0(z_prev, t_idx_s0, src)

    denom = t_s + delta
    u_linear = (delta * dv_s + t_s * dv_s0) / denom
    u_polar = pipe._polar_chord_blend(dv_s, dv_s0, t_s, delta)

    # old global cosine (buggy)
    flat_s = dv_s.reshape(batch, -1)
    flat_s0 = dv_s0.reshape(batch, -1)
    cos_global = (flat_s * flat_s0).sum(1) / (flat_s.norm(1) * flat_s0.norm(1) + 1e-8)

    r_s, e_s = pipe._polar_decompose(dv_s)
    _, e_s0 = pipe._polar_decompose(dv_s0)
    cos_pixel = (e_s * e_s0).sum(dim=1, keepdim=True)

    rel = (u_polar - u_linear).norm() / (u_linear.norm() + 1e-8)
    print(f"||u_polar - u_linear|| / ||u_linear|| = {rel.item():.6f}")
    print(f"global cos(dv_s, dv_s0)           = {cos_global.item():.6f}")
    print(f"per-pixel cos: min={cos_pixel.min().item():.4f} mean={cos_pixel.mean().item():.4f} "
          f"p10={torch.quantile(cos_pixel, 0.1).item():.4f} p50={torch.quantile(cos_pixel, 0.5).item():.4f}")
    print(f"pixels with cos<0.95: {(cos_pixel < 0.95).float().mean().item()*100:.2f}%")


if __name__ == "__main__":
    main()
