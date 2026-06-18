"""Multi-Expert Consultation (MEC) for uncertainty-routed cases.

The implementation follows the paper-level screening--verification--adjudication
workflow while remaining provider-agnostic. Dataset-specific prompts are loaded
from ``prompts/<dataset>/MEC`` through :class:`himec.prompts.PromptBank`.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib import error, request

import torch

from .prompts import PromptBank, PromptBundle, render_prompt


class AgentClient(Protocol):
    def __call__(self, prompt: str, image_path: Optional[str] = None, temperature: float = 0.2) -> str:
        ...


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _extract_chat_content(result: Dict[str, Any]) -> str:
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat completion response: {result}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    return str(content)


class OpenAICompatibleAgent:
    """Synchronous client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4000,
        timeout: int = 60,
        include_image: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("MEC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        resolved_base_url = base_url or os.environ.get("MEC_BASE_URL")
        if not resolved_base_url:
            raise RuntimeError("Missing API base URL. Set MEC_BASE_URL or pass --mec_base_url before running MEC.")
        self.base_url = resolved_base_url.rstrip("/") + "/"
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.include_image = include_image
        if not self.api_key:
            raise RuntimeError("Missing API key. Set MEC_API_KEY or OPENAI_API_KEY before running MEC.")

    def _message_content(self, prompt: str, image_path: Optional[str]) -> Any:
        if image_path and self.include_image:
            return [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
            ]
        return prompt

    def __call__(self, prompt: str, image_path: Optional[str] = None, temperature: float = 0.2) -> str:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": self._message_content(prompt, image_path)}],
        }
        req = request.Request(
            self.base_url + "chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"{self.model} request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"{self.model} request failed: {exc}") from exc
        return _extract_chat_content(result)


class MedGemmaAgent:
    """Local MedGemma vision-language agent using transformers.pipeline."""

    def __init__(
        self,
        model: str = "google/medgemma-1.5-4b-it",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 2000,
    ) -> None:
        try:
            from PIL import Image
            from transformers import pipeline
        except Exception as exc:
            raise ImportError("MedGemmaAgent requires `Pillow` and `transformers`.") from exc
        self.image_cls = Image
        self.pipe = pipeline(
            "image-text-to-text",
            model=model,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str, image_path: Optional[str] = None, temperature: float = 0.2) -> str:
        content = []
        if image_path:
            image = self.image_cls.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        output = self.pipe(
            text=[{"role": "user", "content": content}],
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            last = generated[-1]
            if isinstance(last, dict):
                return str(last.get("content", last))
        return str(generated)


@dataclass
class AgentReport:
    features: List[str]
    reason: str
    diagnosis: str
    confidence: float = 0.0
    raw: str = ""


@dataclass
class AnalystReport:
    disagreement_type: str
    visual_evidence_disagreement: str
    logical_reasoning_disagreement: str
    flaws_of_a: List[str] = field(default_factory=list)
    flaws_of_b: List[str] = field(default_factory=list)
    reliable_evidence: List[str] = field(default_factory=list)
    audit_summary: str = ""
    raw: str = ""


@dataclass
class MECDecision:
    diagnosis: str
    confidence: float
    route: str
    expert_a: Optional[AgentReport] = None
    expert_b: Optional[AgentReport] = None
    analyst: Optional[AnalystReport] = None
    adjudication: Optional[Dict[str, Any]] = None
    reason: str = ""
    router_stats: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class RobustJSON:
    """Parse JSON-like LLM outputs with conservative fallbacks."""

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        diagnosis = ""
        patterns = [
            r"<diagnosis>\s*([^<]+)\s*</diagnosis>",
            r"diagnosis\s*[\":：>]\s*([A-Za-z0-9_\-/ ]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if m:
                diagnosis = m.group(1).strip().strip('"</ ')
                break
        return {"features": [], "reason": text[:1500], "diagnosis": diagnosis, "confidence": 0.0}


class GuidelineStore:
    """Lightweight guideline retrieval with expert-reviewed guideline files."""

    def __init__(self, guideline_root: Optional[str] = None, guideline_json: Optional[str] = None) -> None:
        self.entries: Dict[str, str] = {}
        if guideline_root:
            root = Path(guideline_root).expanduser()
            for p in root.rglob("*.txt"):
                key = p.stem
                if p.parent.name and p.parent.name.lower() != root.name.lower():
                    key = f"{p.parent.name}/{p.stem}"
                self.entries[key] = p.read_text(encoding="utf-8", errors="ignore")
        if guideline_json:
            self.entries.update(json.loads(Path(guideline_json).read_text(encoding="utf-8")))

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[A-Za-z0-9]+", text.lower()))

    def retrieve(self, labels: Sequence[str], dataset: str = "", query: str = "", topk: int = 6) -> str:
        if not self.entries:
            return "No external guideline was provided. Use only visible evidence."
        q = self._tokens(" ".join(labels) + " " + dataset + " " + query)
        scored = []
        label_set = {x.lower() for x in labels}
        for key, text in self.entries.items():
            stem = key.split("/")[-1].lower()
            base = 3.0 if stem in label_set else 0.0
            score = base + len(q & self._tokens(key + " " + text)) / max(1, len(q))
            scored.append((score, key, text))
        scored.sort(reverse=True, key=lambda x: x[0])
        return "\n\n".join([f"[{key}]\n{text.strip()}" for _, key, text in scored[:topk]])


class UncertaintyRouter:
    """Route high-uncertainty samples to MEC.

    By default, this matches the paper setting: route when maximum softmax
    probability is below 0.75. An optional normalized-entropy threshold can be
    enabled for stricter intractable-case detection.
    """

    def __init__(self, threshold: float = 0.75, entropy_threshold: Optional[float] = None) -> None:
        self.threshold = threshold
        self.entropy_threshold = entropy_threshold

    def should_route(self, logits: torch.Tensor) -> Tuple[bool, Dict[str, float]]:
        probs = torch.softmax(logits.float(), dim=-1)
        max_prob = float(probs.max().item())
        pred_idx = int(probs.argmax().item())
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item() / math.log(probs.numel()))
        margin = float((torch.topk(probs, k=min(2, probs.numel())).values[0] - torch.topk(probs, k=min(2, probs.numel())).values[-1]).item()) if probs.numel() > 1 else 1.0
        route = max_prob < self.threshold
        if self.entropy_threshold is not None:
            route = route or entropy > self.entropy_threshold
        return route, {"max_prob": max_prob, "entropy": entropy, "margin": margin, "pred_idx": float(pred_idx)}


class TraceWriter:
    """Append MEC traces as JSONL for reproducible debugging."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class MultiExpertConsultation:
    def __init__(
        self,
        expert_a: AgentClient,
        expert_b: AgentClient,
        analyst: AgentClient,
        adjudicator: AgentClient,
        guideline_store: Optional[GuidelineStore] = None,
        router: Optional[UncertaintyRouter] = None,
        prompt_bundle: Optional[PromptBundle] = None,
        prompt_root: Optional[str] = None,
        dataset: str = "",
        expert_temperature: float = 0.7,
        trace_path: Optional[str] = None,
    ) -> None:
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.analyst = analyst
        self.adjudicator = adjudicator
        self.guidelines = guideline_store or GuidelineStore()
        self.router = router or UncertaintyRouter(0.75)
        self.dataset = dataset
        self.prompt_bundle = prompt_bundle or PromptBank(prompt_root).load(dataset)
        self.expert_temperature = expert_temperature
        self.trace = TraceWriter(trace_path)

    @staticmethod
    def _normalize_diagnosis(label: str, labels: Sequence[str]) -> str:
        label_clean = str(label).strip()
        if label_clean in labels:
            return label_clean
        lower_map = {x.lower(): x for x in labels}
        if label_clean.lower() in lower_map:
            return lower_map[label_clean.lower()]
        for cand in labels:
            if cand.lower() in label_clean.lower():
                return cand
        return label_clean

    @staticmethod
    def _report_from_raw(raw: str, labels: Sequence[str]) -> AgentReport:
        data = RobustJSON.parse(raw)
        features = data.get("features", [])
        if isinstance(features, str):
            features = [features]
        diagnosis = MultiExpertConsultation._normalize_diagnosis(str(data.get("diagnosis", "")), labels)
        return AgentReport(
            features=[str(x) for x in features],
            reason=str(data.get("reason", data.get("rationale", ""))),
            diagnosis=diagnosis,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            raw=raw,
        )

    @staticmethod
    def _analyst_from_raw(raw: str) -> AnalystReport:
        data = RobustJSON.parse(raw)
        def as_list(x: Any) -> List[str]:
            if x is None:
                return []
            if isinstance(x, list):
                return [str(v) for v in x]
            return [str(x)]
        return AnalystReport(
            disagreement_type=str(data.get("disagreement_type", "mixed")),
            visual_evidence_disagreement=str(data.get("visual_evidence_disagreement", data.get("evidence_dis", ""))),
            logical_reasoning_disagreement=str(data.get("logical_reasoning_disagreement", data.get("logic_dis", ""))),
            flaws_of_a=as_list(data.get("flaws_of_a", data.get("flaw_of_a", []))),
            flaws_of_b=as_list(data.get("flaws_of_b", data.get("flaw_of_b", []))),
            reliable_evidence=as_list(data.get("reliable_evidence", [])),
            audit_summary=str(data.get("audit_summary", data.get("Flaw", data.get("flaw", "")))),
            raw=raw,
        )

    def _stage1(self, image_path: str, labels: Sequence[str], guideline: str, base_prediction: str, base_confidence: float) -> Tuple[AgentReport, AgentReport]:
        ctx = {
            "image_path": image_path,
            "labels": ", ".join(labels),
            "guideline": guideline,
            "base_prediction": base_prediction,
            "base_confidence": f"{base_confidence:.4f}",
        }
        prompt = render_prompt(self.prompt_bundle.stage1, ctx, fallback_context_title="Hi-MEC Stage-1 Context")
        raw_a = self.expert_a(prompt, image_path=image_path, temperature=self.expert_temperature)
        raw_b = self.expert_b(prompt, image_path=image_path, temperature=self.expert_temperature)
        return self._report_from_raw(raw_a, labels), self._report_from_raw(raw_b, labels)

    def _stage2(self, report_a: AgentReport, report_b: AgentReport, guideline: str) -> AnalystReport:
        ctx = {
            "report_a": json.dumps(asdict(report_a), ensure_ascii=False, indent=2),
            "report_b": json.dumps(asdict(report_b), ensure_ascii=False, indent=2),
            "guideline": guideline,
        }
        prompt = render_prompt(self.prompt_bundle.stage2, ctx, fallback_context_title="Hi-MEC Stage-2 Context")
        raw = self.analyst(prompt, image_path=None, temperature=0.2)
        return self._analyst_from_raw(raw)

    def _stage3(self, image_path: str, report_a: AgentReport, report_b: AgentReport, analyst: AnalystReport, labels: Sequence[str], guideline: str) -> Dict[str, Any]:
        ctx = {
            "image_path": image_path,
            "labels": ", ".join(labels),
            "report_a": json.dumps(asdict(report_a), ensure_ascii=False, indent=2),
            "report_b": json.dumps(asdict(report_b), ensure_ascii=False, indent=2),
            "flaw_report": json.dumps(asdict(analyst), ensure_ascii=False, indent=2),
            "guideline": guideline,
        }
        prompt = render_prompt(self.prompt_bundle.stage3, ctx, fallback_context_title="Hi-MEC Stage-3 Context")
        raw = self.adjudicator(prompt, image_path=image_path, temperature=0.1)
        data = RobustJSON.parse(raw)
        data["raw"] = raw
        data["diagnosis"] = self._normalize_diagnosis(str(data.get("diagnosis", "")), labels)
        return data

    def consult(
        self,
        image_path: str,
        base_logits: torch.Tensor,
        labels: Sequence[str],
        base_prediction: Optional[str] = None,
        query: str = "",
        force: bool = False,
    ) -> MECDecision:
        logits = base_logits.detach().cpu().view(-1)
        route, stats = self.router.should_route(logits)
        if base_prediction is None:
            base_prediction = labels[int(torch.argmax(logits).item())]
        if not route and not force:
            decision = MECDecision(
                diagnosis=base_prediction,
                confidence=stats["max_prob"],
                route="base",
                reason=f"Not routed to MEC: max_prob={stats['max_prob']:.3f}, entropy={stats['entropy']:.3f}.",
                router_stats=stats,
            )
            self.trace.write({"image_path": image_path, "decision": decision.to_json()})
            return decision

        guideline = self.guidelines.retrieve(labels, dataset=self.dataset, query=query)
        a, b = self._stage1(image_path, labels, guideline, base_prediction, stats["max_prob"])
        if a.diagnosis and a.diagnosis == b.diagnosis:
            conf = max(a.confidence, b.confidence, stats["max_prob"])
            decision = MECDecision(
                diagnosis=a.diagnosis,
                confidence=conf,
                route="stage1_consensus",
                expert_a=a,
                expert_b=b,
                reason="Two independent experts reached consensus.",
                router_stats=stats,
            )
            self.trace.write({"image_path": image_path, "decision": decision.to_json()})
            return decision

        analyst = self._stage2(a, b, guideline)
        final = self._stage3(image_path, a, b, analyst, labels, guideline)
        diagnosis = str(final.get("diagnosis", "")).strip() or base_prediction
        confidence = float(final.get("confidence", stats["max_prob"]) or stats["max_prob"])
        decision = MECDecision(
            diagnosis=diagnosis,
            confidence=confidence,
            route="stage3_adjudication",
            expert_a=a,
            expert_b=b,
            analyst=analyst,
            adjudication=final,
            reason=str(final.get("decision_rationale", final.get("reason", ""))),
            router_stats=stats,
        )
        self.trace.write({"image_path": image_path, "decision": decision.to_json()})
        return decision


class EchoAgent:
    """Debug-only agent returning deterministic JSON.

    This lets `scripts/test_himec.py --with_mec --mock_mec` verify the full MEC
    call chain without external APIs.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label

    def __call__(self, prompt: str, image_path: Optional[str] = None, temperature: float = 0.2) -> str:
        label = self.label or ""
        if "Disagreement Analyst" in prompt or "Stage-2" in prompt:
            return json.dumps({
                "disagreement_type": "mixed",
                "visual_evidence_disagreement": "mock visual disagreement",
                "logical_reasoning_disagreement": "mock logical disagreement",
                "flaws_of_a": [],
                "flaws_of_b": [],
                "reliable_evidence": [],
                "audit_summary": "debug analyst",
            })
        return json.dumps({"features": [], "reason": "debug agent", "diagnosis": label, "confidence": 0.5})
