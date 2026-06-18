"""Hi-MEC neural backbone.

The model implements the trainable Base stage of Hi-MEC:
  MedMamba visual encoding -> BioClinical-BERT attribute encoding ->
  QACDF with BioMedCLIP scores -> reciprocal semantic-visual alignment ->
  entropy-gated adaptive fusion.

MEC is intentionally implemented in `himec.mec` because it is activated only
for high-uncertainty samples during inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .backbones import build_medmamba
from .modules import EntropyGatedAdaptiveFusion, MaskedMeanPooling, QACDF, ReciprocalSemanticVisualAlign


@dataclass
class HiMECConfig:
    num_classes: int
    hidden_dim: int = 768
    num_attributes: int = 5
    text_encoder_path: str = "emilyalsentzer/Bio_ClinicalBERT"
    medmamba_checkpoint: Optional[str] = None
    freeze_text_encoder: bool = True
    freeze_visual_encoder: bool = False
    qacdf_temperature: float = 0.07
    align_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1
    aux_loss_weight: float = 0.1
    debug_tiny_visual: bool = False


class HiMECBase(nn.Module):
    def __init__(self, cfg: HiMECConfig) -> None:
        super().__init__()
        self.cfg = cfg
        C = cfg.hidden_dim
        self.visual_encoder = build_medmamba(C, cfg.medmamba_checkpoint, fallback=cfg.debug_tiny_visual)
        self.text_encoder = AutoModel.from_pretrained(cfg.text_encoder_path)
        self.pool_text = MaskedMeanPooling()
        if cfg.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        if cfg.freeze_visual_encoder:
            for p in self.visual_encoder.parameters():
                p.requires_grad_(False)

        self.qacdf = QACDF(C, temperature=cfg.qacdf_temperature, dropout=cfg.dropout)
        self.align = ReciprocalSemanticVisualAlign(C, cfg.num_heads, cfg.dropout, cfg.align_layers)
        self.eaf = EntropyGatedAdaptiveFusion(C, cfg.num_classes, cfg.dropout, cfg.aux_loss_weight)

    def encode_visual(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_hw = self.visual_encoder.forward_backbone(image)  # [B,H,W,C]
        B, H, W, C = feat_hw.shape
        tokens = feat_hw.view(B, H * W, C)
        return {"tokens": tokens, "global": tokens.mean(dim=1)}

    def encode_attributes(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # input_ids: [B,K,S]
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
        B, K, S = input_ids.shape
        outputs = self.text_encoder(
            input_ids=input_ids.reshape(B * K, S),
            attention_mask=attention_mask.reshape(B * K, S),
        )
        pooled = self.pool_text(outputs.last_hidden_state, attention_mask.reshape(B * K, S))
        pooled = pooled.view(B, K, -1)
        # use one token per attribute, which is exactly the local-knowledge token sequence
        return {"attr_feat": pooled, "tokens": pooled}

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        biomedclip_scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        visual = self.encode_visual(image)
        text = self.encode_attributes(input_ids, attention_mask)

        qout = self.qacdf(text["attr_feat"], biomedclip_scores)
        # append the QACDF-fused token to preserve global textual context
        text_tokens = torch.cat([text["tokens"], qout["fused_text"].unsqueeze(1)], dim=1)
        aligned = self.align(visual["tokens"], text_tokens)
        eaf_out = self.eaf(
            f_v=aligned["visual_enhanced"],
            f_t=aligned["text_enhanced"],
            f_g=visual["global"],
            labels=labels,
        )
        out = {
            **eaf_out,
            "attr_weights": qout["attr_weights"],
            "attr_logits": qout["attr_logits"],
            "visual_global": visual["global"],
            "text_fused": qout["fused_text"],
        }
        return out if return_dict else out["logits"]
