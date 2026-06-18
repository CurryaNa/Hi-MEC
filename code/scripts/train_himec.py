from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from himec.biomedclip import BioMedCLIPConfig, BioMedCLIPScorer
from himec.data import HiMECDataset, himec_collate
from himec.model import HiMECBase, HiMECConfig
from himec.train import evaluate, train_one_epoch


def parse_args():
    p = argparse.ArgumentParser("Train Hi-MEC Base")
    p.add_argument("--data_root", required=True)
    p.add_argument("--text_encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--medmamba_checkpoint", default=None)
    p.add_argument("--biomedclip_dir", default="checkpoints/biomedclip")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", default="runs/himec")
    p.add_argument("--debug_tiny_visual", action="store_true")
    p.add_argument("--disable_biomedclip", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = HiMECDataset(args.data_root, "train", args.text_encoder)
    val_set = HiMECDataset(args.data_root, "val", args.text_encoder, class_names=train_set.class_names)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=himec_collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=himec_collate, pin_memory=True)

    cfg = HiMECConfig(
        num_classes=args.num_classes,
        text_encoder_path=args.text_encoder,
        medmamba_checkpoint=args.medmamba_checkpoint,
        debug_tiny_visual=args.debug_tiny_visual,
    )
    model = HiMECBase(cfg).to(device)
    biomedclip = None if args.disable_biomedclip else BioMedCLIPScorer(BioMedCLIPConfig(local_dir=args.biomedclip_dir, device=device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, biomedclip)
        va = evaluate(model, val_loader, device, biomedclip)
        print(f"Epoch {epoch:03d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | val acc {va['acc']:.4f}")
        if va["acc"] > best:
            best = va["acc"]
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "class_names": train_set.class_names}, save_dir / "best.pt")


if __name__ == "__main__":
    main()
