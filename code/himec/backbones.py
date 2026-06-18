"""Backbone helpers.

The default visual encoder is MedMamba when available. A tiny CNN fallback is
provided only for debugging the code path without the external MedMamba file.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class TinyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, output_dim, 3, stride=2, padding=1), nn.BatchNorm2d(output_dim), nn.GELU(),
        )

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # return [B, H, W, C] to mimic VSSM.forward_backbone
        return self.net(x).permute(0, 2, 3, 1).contiguous()


def build_medmamba(output_dim: int = 768, checkpoint_path: Optional[str] = None, fallback: bool = False) -> nn.Module:
    if fallback:
        return TinyVisualEncoder(output_dim)
    try:
        from .MedMamba import VSSM
        model = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, output_dim], num_classes=0)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        return model
    except Exception as exc:
        raise ImportError(
            "Failed to build MedMamba. Set fallback=True for debugging or provide the MedMamba dependencies. "
            f"Original error: {exc}"
        )
