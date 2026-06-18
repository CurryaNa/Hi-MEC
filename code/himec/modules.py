"""Core neural modules of Hi-MEC: QACDF, reciprocal alignment, and EAF."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMeanPooling(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class QACDF(nn.Module):
    """Quality-Aware Cross-Scale Dynamic Fusion.

    It takes task-specific text embeddings from BioClinical-BERT and quality
    scores from BioMedCLIP. The scores are used only as reliability weights,
    which keeps reliability estimation decoupled from representation learning.
    """

    def __init__(self, hidden_dim: int = 768, temperature: float = 0.07, dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.local_refiner = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_attr_feat: torch.Tensor, biomedclip_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parameters
        text_attr_feat: [B, K, C] BioClinical-BERT attribute features.
        biomedclip_scores: [B, K] cosine scores from BioMedCLIP.
        """
        if text_attr_feat.dim() != 3:
            raise ValueError("text_attr_feat should have shape [B, K, C]")
        if biomedclip_scores.shape[:2] != text_attr_feat.shape[:2]:
            raise ValueError("biomedclip_scores should have shape [B, K]")

        refined = self.local_refiner(text_attr_feat) + text_attr_feat
        q = biomedclip_scores.unsqueeze(-1)
        learned_residual = self.quality_gate(torch.cat([refined, q], dim=-1)).squeeze(-1)
        logits = biomedclip_scores / max(self.temperature, 1e-6) + learned_residual
        weights = torch.softmax(logits, dim=1)
        fused = torch.einsum("bk,bkc->bc", weights, refined)
        return {
            "fused_text": self.output_norm(fused),
            "attr_weights": weights,
            "attr_logits": logits,
            "refined_attr_feat": refined,
        }


class ReciprocalSemanticVisualAlign(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_heads: int = 4, dropout: float = 0.1, n_layers: int = 1) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.text_self = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.v2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.t2v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t1 = nn.LayerNorm(hidden_dim)
        self.norm_t2 = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.ffn_t = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.ffn_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        v, t = visual_tokens, text_tokens
        for _ in range(self.n_layers):
            t_sa, _ = self.text_self(t, t, t, need_weights=False)
            t = self.norm_t1(t + t_sa)
            t_ca, _ = self.v2t(t, v, v, need_weights=False)
            t = self.norm_t2(t + t_ca + self.ffn_t(t))
            v_ca, _ = self.t2v(v, t, t, need_weights=False)
            v = self.norm_v(v + v_ca + self.ffn_v(v))
        return {
            "visual_tokens": v,
            "text_tokens": t,
            "visual_enhanced": v.mean(dim=1),
            "text_enhanced": t.mean(dim=1),
        }


class EntropyGatedAdaptiveFusion(nn.Module):
    """Entropy-Gated Modality Adaptive Fusion.

    Three branches are used: enhanced visual, enhanced textual, and global visual.
    Branch-wise auxiliary classifiers provide reliable entropy estimates; weights
    are computed by negative-entropy softmax.
    """

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.2, aux_loss_weight: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.aux_loss_weight = aux_loss_weight
        self.proj_v = self._projector(hidden_dim, dropout)
        self.proj_t = self._projector(hidden_dim, dropout)
        self.proj_g = self._projector(hidden_dim, dropout)
        self.cls_v = nn.Linear(hidden_dim, num_classes)
        self.cls_t = nn.Linear(hidden_dim, num_classes)
        self.cls_g = nn.Linear(hidden_dim, num_classes)
        self.final_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _projector(dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def entropy(self, logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = torch.softmax(logits, dim=-1).clamp_min(eps)
        return -(p * p.log()).sum(dim=-1) / math.log(self.num_classes)

    def forward(self, f_v: torch.Tensor, f_t: torch.Tensor, f_g: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        z_v, z_t, z_g = self.proj_v(f_v), self.proj_t(f_t), self.proj_g(f_g)
        logit_v, logit_t, logit_g = self.cls_v(z_v), self.cls_t(z_t), self.cls_g(z_g)
        ent = torch.stack([self.entropy(logit_v), self.entropy(logit_t), self.entropy(logit_g)], dim=1)
        weights = torch.softmax(-ent, dim=1).detach()  # detach stabilizes early training
        fused = weights[:, 0:1] * z_v + weights[:, 1:2] * z_t + weights[:, 2:3] * z_g
        logits = self.final_classifier(fused)
        out = {
            "logits": logits,
            "fused_feature": fused,
            "modality_weights": weights,
            "entropy": ent,
            "branch_logits": {"visual": logit_v, "text": logit_t, "global": logit_g},
        }
        if labels is not None:
            main = F.cross_entropy(logits, labels)
            aux = F.cross_entropy(logit_v, labels) + F.cross_entropy(logit_t, labels) + F.cross_entropy(logit_g, labels)
            out["loss"] = main + self.aux_loss_weight * aux
            out["loss_main"] = main.detach()
            out["loss_aux"] = aux.detach()
        return out
