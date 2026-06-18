from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from himec.biomedclip import BioMedCLIPConfig, BioMedCLIPScorer
from himec.data import HiMECDataset, himec_collate
from himec.mec import EchoAgent, GuidelineStore, MedGemmaAgent, MultiExpertConsultation, OpenAICompatibleAgent, UncertaintyRouter
from himec.model import HiMECBase, HiMECConfig
from himec.train import predict


def parse_args():
    p = argparse.ArgumentParser("Test Hi-MEC Base / Full")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", required=True, help="Path to runs/himec/best.pt")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--text_encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--medmamba_checkpoint", default=None)
    p.add_argument("--biomedclip_dir", default="checkpoints/biomedclip")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_jsonl", default="runs/himec/test_predictions.jsonl")
    p.add_argument("--debug_tiny_visual", action="store_true")
    p.add_argument("--disable_biomedclip", action="store_true")

    # Full Hi-MEC / MEC options
    p.add_argument("--with_mec", action="store_true", help="Run uncertainty-routed MEC after base inference.")
    p.add_argument("--mock_mec", action="store_true", help="Use EchoAgent to verify MEC pipeline without external APIs.")
    p.add_argument("--prompt_root", default="prompts", help="Root containing dataset-specific prompts.")
    p.add_argument("--dataset_name", default="", help="Prompt subfolder name, e.g. breastmnist, DermaMNIST, PAD-UFES-20.")
    p.add_argument("--guideline_root", default=None)
    p.add_argument("--mec_threshold", type=float, default=0.75)
    p.add_argument("--force_mec", action="store_true")
    p.add_argument("--mec_base_url", default=None, help="OpenAI-compatible base URL for GPT/Gemini agents. Prefer MEC_BASE_URL env var.")
    p.add_argument("--mec_api_key", default=None, help="API key. Prefer MEC_API_KEY/OPENAI_API_KEY env vars instead of this flag.")
    p.add_argument("--medgemma_model", default="google/medgemma-1.5-4b-it")
    p.add_argument("--medgemma_device", default="cuda")
    p.add_argument("--medgemma_max_new_tokens", type=int, default=2000)
    p.add_argument("--mec_expert_temperature", type=float, default=1.0, help="High-temperature sampling for the two independent MedGemma stage-1 reports.")
    p.add_argument("--gpt_model", default="gpt-5.1")
    p.add_argument("--gemini_model", default="gemini-3-pro-preview")
    p.add_argument("--mec_max_tokens", type=int, default=4000)
    p.add_argument("--mec_timeout", type=int, default=60)
    return p.parse_args()


def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    vals = []
    for c in range(num_classes):
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == c and p == c)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != c and p == c)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == c and p != c)
        denom = 2 * tp + fp + fn
        vals.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(vals) / max(len(vals), 1)


def load_model(args, device: str) -> HiMECBase:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict: Dict[str, object] = ckpt.get("cfg", {})
    cfg = HiMECConfig(
        num_classes=args.num_classes,
        text_encoder_path=args.text_encoder or str(cfg_dict.get("text_encoder_path", "emilyalsentzer/Bio_ClinicalBERT")),
        medmamba_checkpoint=args.medmamba_checkpoint or cfg_dict.get("medmamba_checkpoint"),
        debug_tiny_visual=args.debug_tiny_visual or bool(cfg_dict.get("debug_tiny_visual", False)),
    )
    model = HiMECBase(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_names = ckpt.get("class_names")

    dataset = HiMECDataset(args.data_root, args.split, args.text_encoder, class_names=class_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=himec_collate, pin_memory=True)
    model = load_model(args, device)
    biomedclip = None if args.disable_biomedclip else BioMedCLIPScorer(BioMedCLIPConfig(local_dir=args.biomedclip_dir, device=device))

    records = predict(model, loader, device, biomedclip)
    y_true = [int(r["label"]) for r in records]
    y_pred = [int(r["pred"]) for r in records]
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    f1 = macro_f1(y_true, y_pred, args.num_classes)

    labels = dataset.class_names
    if args.with_mec:
        fallback_label = labels[0] if labels else ""
        second_label = labels[1] if len(labels) > 1 else fallback_label
        if args.mock_mec:
            expert_a = EchoAgent(fallback_label)
            expert_b = EchoAgent(second_label)
            analyst = EchoAgent()
            adjudicator = EchoAgent(fallback_label)
        else:
            medgemma = MedGemmaAgent(
                model=args.medgemma_model,
                device=args.medgemma_device,
                max_new_tokens=args.medgemma_max_new_tokens,
            )
            expert_a = medgemma
            expert_b = medgemma
            analyst = OpenAICompatibleAgent(
                model=args.gpt_model,
                api_key=args.mec_api_key,
                base_url=args.mec_base_url,
                max_tokens=args.mec_max_tokens,
                timeout=args.mec_timeout,
                include_image=False,
            )
            adjudicator = OpenAICompatibleAgent(
                model=args.gemini_model,
                api_key=args.mec_api_key,
                base_url=args.mec_base_url,
                max_tokens=args.mec_max_tokens,
                timeout=args.mec_timeout,
                include_image=True,
            )
        mec = MultiExpertConsultation(
            expert_a=expert_a,
            expert_b=expert_b,
            analyst=analyst,
            adjudicator=adjudicator,
            guideline_store=GuidelineStore(guideline_root=args.guideline_root),
            router=UncertaintyRouter(threshold=args.mec_threshold),
            prompt_root=args.prompt_root,
            dataset=args.dataset_name or Path(args.data_root).name,
            expert_temperature=args.mec_expert_temperature,
            trace_path=str(Path(args.save_jsonl).with_suffix(".mec_trace.jsonl")),
        )
        routed = 0
        mec_pred = []
        for r in records:
            base_label = labels[int(r["pred"])]
            decision = mec.consult(
                image_path=str(r["image_path"]),
                base_logits=torch.tensor(r["logits"]),
                labels=labels,
                base_prediction=base_label,
                query=str(r.get("class_name", "")),
                force=args.force_mec,
            )
            r["mec"] = decision.to_json()
            routed += int(decision.route != "base")
            mec_label = decision.diagnosis if decision.diagnosis in labels else base_label
            mec_pred.append(labels.index(mec_label))
        mec_acc = sum(int(a == b) for a, b in zip(y_true, mec_pred)) / max(len(y_true), 1)
        mec_f1 = macro_f1(y_true, mec_pred, args.num_classes)
        print(f"Base OA={acc:.4f} macro-F1={f1:.4f} | Hi-MEC Full OA={mec_acc:.4f} macro-F1={mec_f1:.4f} | routed={routed}/{len(records)}")
    else:
        print(f"Hi-MEC Base OA={acc:.4f} macro-F1={f1:.4f} | N={len(records)}")

    out_path = Path(args.save_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
