"""Dataset utilities for Hi-MEC.

Expected layout:
  root/
    image/{train,val,test}/{class_name}/*.png|jpg|jpeg|tif
    text/{train,val,test}/{class_name}/*.txt

Each text file may contain either a global description or K lines of
attribute-level knowledge. The loader always returns a fixed-length list of
attributes, padding missing fields with empty strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


@dataclass
class Sample:
    image_path: str
    text_path: str
    label: int
    class_name: str


class HiMECDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        tokenizer_path: str,
        num_attributes: int = 5,
        max_length: int = 256,
        image_size: int = 224,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_attributes = num_attributes
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.samples, self.class_names = self._scan(class_names)

    def _scan(self, class_names: Optional[Sequence[str]]) -> Tuple[List[Sample], List[str]]:
        image_root = self.root / "image" / self.split
        text_root = self.root / "text" / self.split
        if not image_root.exists():
            raise FileNotFoundError(f"Image split directory not found: {image_root}")
        names = list(class_names) if class_names is not None else sorted([p.name for p in image_root.iterdir() if p.is_dir()])
        samples: List[Sample] = []
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        for label, cls in enumerate(names):
            img_dir = image_root / cls
            txt_dir = text_root / cls
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in exts:
                    continue
                txt_path = txt_dir / f"{img_path.stem}.txt"
                if txt_path.exists():
                    samples.append(Sample(str(img_path), str(txt_path), label, cls))
        if len(samples) == 0:
            raise RuntimeError(f"No paired image/text samples found under {self.root} [{self.split}]")
        return samples, names

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _strip_numbering(line: str) -> str:
        line = re.sub(r"^\s*[-*•]?\s*\d+\s*[\.)：:]\s*", "", line.strip())
        return line.strip()

    def _read_attributes(self, path: str) -> List[str]:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        lines = [self._strip_numbering(x) for x in raw.splitlines() if x.strip()]
        if len(lines) == 1 and ";" in lines[0]:
            lines = [self._strip_numbering(x) for x in lines[0].split(";") if x.strip()]
        lines = lines[: self.num_attributes]
        lines += [""] * (self.num_attributes - len(lines))
        return lines

    def __getitem__(self, index: int) -> Dict[str, object]:
        s = self.samples[index]
        image = self.transform(Image.open(s.image_path).convert("RGB"))
        attrs = self._read_attributes(s.text_path)
        encoded = self.tokenizer(
            attrs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "image": image,
            "input_ids": encoded["input_ids"],          # [K, S]
            "attention_mask": encoded["attention_mask"],# [K, S]
            "attributes": attrs,
            "label": torch.tensor(s.label, dtype=torch.long),
            "image_path": s.image_path,
            "text_path": s.text_path,
            "class_name": s.class_name,
        }


def himec_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "attributes": [x["attributes"] for x in batch],
        "label": torch.stack([x["label"] for x in batch], dim=0),
        "image_path": [x["image_path"] for x in batch],
        "text_path": [x["text_path"] for x in batch],
        "class_name": [x["class_name"] for x in batch],
    }
