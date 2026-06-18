"""BioMedCLIP wrapper for QACDF.

This module follows the official OpenCLIP-style loading procedure for
`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` and exposes a
small interface tailored to Hi-MEC: image/text encoding and attribute-level
visual support scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
except Exception as exc:  # pragma: no cover
    hf_hub_download = None
    create_model_and_transforms = None
    get_tokenizer = None
    HF_HUB_PREFIX = "hf-hub:"
    _MODEL_CONFIGS = {}
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@dataclass(frozen=True)
class BioMedCLIPConfig:
    repo_id: str = BIOMEDCLIP_REPO
    local_dir: str = "checkpoints/biomedclip"
    model_name: str = "biomedclip_local"
    context_length: int = 256
    device: Optional[str] = None
    freeze: bool = True


class BioMedCLIPScorer:
    """Frozen biomedical image-text evaluator used by QACDF.

    The scorer is intentionally decoupled from the task-specific encoders:
    MedMamba/BioClinical-BERT learn representations for diagnosis, whereas
    BioMedCLIP only estimates whether each textual attribute is visually
    supported by the input image.
    """

    def __init__(self, cfg: BioMedCLIPConfig = BioMedCLIPConfig()) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "BioMedCLIPScorer requires `open_clip_torch` and `huggingface_hub`. "
                f"Original import error: {_IMPORT_ERROR}"
            )
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.local_dir = Path(cfg.local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_checkpoint()
        self.model, self.preprocess, self.tokenizer = self._load_local_model()
        self.model.to(self.device).eval()
        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def _ensure_checkpoint(self) -> None:
        model_file = self.local_dir / "open_clip_pytorch_model.bin"
        config_file = self.local_dir / "open_clip_config.json"
        if not model_file.exists():
            hf_hub_download(self.cfg.repo_id, "open_clip_pytorch_model.bin", local_dir=str(self.local_dir))
        if not config_file.exists():
            hf_hub_download(self.cfg.repo_id, "open_clip_config.json", local_dir=str(self.local_dir))

    def _load_local_model(self):
        with open(self.local_dir / "open_clip_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]
        if (
            not self.cfg.model_name.startswith(HF_HUB_PREFIX)
            and self.cfg.model_name not in _MODEL_CONFIGS
        ):
            _MODEL_CONFIGS[self.cfg.model_name] = model_cfg
        tokenizer = get_tokenizer(self.cfg.model_name)
        model, _, preprocess = create_model_and_transforms(
            model_name=self.cfg.model_name,
            pretrained=str(self.local_dir / "open_clip_pytorch_model.bin"),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        return model, preprocess, tokenizer

    @torch.no_grad()
    def encode_images(self, images: Union[torch.Tensor, Sequence[Union[str, Path, Image.Image]]]) -> torch.Tensor:
        """Return L2-normalized image embeddings of shape [B, D]."""
        if isinstance(images, torch.Tensor):
            image_tensor = images.to(self.device)
        else:
            processed = []
            for item in images:
                img = item if isinstance(item, Image.Image) else Image.open(item).convert("RGB")
                processed.append(self.preprocess(img))
            image_tensor = torch.stack(processed, dim=0).to(self.device)
        feat = self.model.encode_image(image_tensor)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Return L2-normalized text embeddings of shape [N, D]."""
        tokens = self.tokenizer(list(texts), context_length=self.cfg.context_length).to(self.device)
        feat = self.model.encode_text(tokens)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def score_attributes(
        self,
        images: Union[torch.Tensor, Sequence[Union[str, Path, Image.Image]]],
        attributes: Sequence[Sequence[str]],
        normalize: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute image-attribute support scores.

        Parameters
        ----------
        images:
            Batch of images or paths, length B.
        attributes:
            Nested list with shape [B, K]. Each element is one attribute-level
            sentence generated by the two-stage CoT prompt.
        normalize:
            If True, return softmax-normalized QACDF weights; otherwise return
            raw cosine similarities.
        temperature:
            Temperature for softmax when ``normalize=True``.
        """
        if len(attributes) == 0:
            raise ValueError("attributes must be a non-empty nested sequence")
        batch = len(attributes)
        num_attr = len(attributes[0])
        if any(len(x) != num_attr for x in attributes):
            raise ValueError("all samples must have the same number of attributes")

        image_feat = self.encode_images(images)  # [B, D]
        flat_text = [t for sample in attributes for t in sample]
        text_feat = self.encode_texts(flat_text).view(batch, num_attr, -1)  # [B, K, D]
        sim = torch.einsum("bd,bkd->bk", image_feat, text_feat)
        if normalize:
            return torch.softmax(sim / max(temperature, 1e-6), dim=-1)
        return sim
