from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .biomedclip import BioMedCLIPScorer
from .model import HiMECBase


def _get_biomedclip_scores(batch: Dict[str, object], device: str, biomedclip: Optional[BioMedCLIPScorer]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    if biomedclip is None:
        return torch.zeros(input_ids.shape[:2], device=device)
    return biomedclip.score_attributes(batch["image_path"], batch["attributes"], normalize=False).to(device)


def train_one_epoch(model: HiMECBase, loader: DataLoader, optimizer, device: str, biomedclip: Optional[BioMedCLIPScorer] = None) -> Dict[str, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scores = _get_biomedclip_scores(batch, device, biomedclip)
        out = model(image, input_ids, attention_mask, scores, labels=labels)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_correct += int((out["logits"].argmax(dim=-1) == labels).sum().item())
        total += labels.numel()
    return {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}


@torch.no_grad()
def predict(model: HiMECBase, loader: DataLoader, device: str, biomedclip: Optional[BioMedCLIPScorer] = None) -> List[Dict[str, object]]:
    model.eval()
    records: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scores = _get_biomedclip_scores(batch, device, biomedclip)
        out = model(image, input_ids, attention_mask, scores, labels=None)
        probs = torch.softmax(out["logits"], dim=-1)
        for i in range(labels.numel()):
            records.append({
                "image_path": batch["image_path"][i],
                "text_path": batch["text_path"][i],
                "class_name": batch["class_name"][i],
                "label": int(labels[i].item()),
                "logits": out["logits"][i].detach().cpu().tolist(),
                "probs": probs[i].detach().cpu().tolist(),
                "pred": int(probs[i].argmax().item()),
                "attr_weights": out["attr_weights"][i].detach().cpu().tolist(),
                "modality_weights": out["modality_weights"][i].detach().cpu().tolist(),
                "entropy": out["entropy"][i].detach().cpu().tolist(),
            })
    return records


@torch.no_grad()
def evaluate(model: HiMECBase, loader: DataLoader, device: str, biomedclip: Optional[BioMedCLIPScorer] = None) -> Dict[str, float]:
    records = predict(model, loader, device, biomedclip)
    total = len(records)
    correct = sum(int(r["pred"] == r["label"]) for r in records)
    return {"acc": correct / max(total, 1)}
