"""Amplified Conditional Transport (ACT) for one-step image editing."""

from __future__ import annotations

import torch

DEFAULT_ACT_AMPLIFICATION = 2.0


def ACT(
    x_anchor: torch.Tensor,
    peak_src: torch.Tensor,
    peak_tgt: torch.Tensor,
    amplification: float = DEFAULT_ACT_AMPLIFICATION,
) -> torch.Tensor:
    """
    Amplified Conditional Transport field.

    With default amplification k=2 this reduces to:
        u = 3 * peak_tgt - peak_src - x_anchor
    """
    k = float(amplification)
    return (1.0 + k) * peak_tgt - peak_src - x_anchor
