"""Prompt loading and templates for Hi-MEC.

This module supports two use cases:
1. Built-in paper-aligned fallback prompts, useful for releasing runnable code.
2. Dataset-specific prompt files stored outside the code directory, e.g.::

    prompts/
      breastmnist/
        br_cot.txt
        MEC/br_stage1.txt
        MEC/br_stage2.txt
        MEC/br_stage3.txt
      DermaMNIST/
        der_cot.txt
        MEC/der_stage1.txt
        MEC/der_stage2.txt
        MEC/der_stage3.txt

The MEC controller reads these files through :class:`PromptBank` and injects
runtime context without overwriting your manually written prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence


TWO_STAGE_COT_TEMPLATE = """[Role]
You are a medical imaging expert. Generate image-grounded visual attributes for the given case.

[Stage 1: Objective Visual Observation]
Describe only directly observable visual evidence. Do not provide a diagnosis, malignancy judgment,
or unsupported clinical history.

[Stage 2: Attribute-level Knowledge Consolidation]
Based on the observation, summarize the image using the following dimensions:
{dimensions}

[Output Format]
<objective_observation>
...
</objective_observation>
<attributes>
{attribute_slots}
</attributes>
"""


def build_two_stage_cot_prompt(dimensions: Sequence[str]) -> str:
    numbered = "\n".join([f"{i + 1}. {d}" for i, d in enumerate(dimensions)])
    slots = "\n".join([f"{i + 1}. {d}: ..." for i, d in enumerate(dimensions)])
    return TWO_STAGE_COT_TEMPLATE.format(dimensions=numbered, attribute_slots=slots)


EXPERT_DIAGNOSIS_PROMPT = """[Role]
You are an independent medical imaging expert.

[Task]
Analyze the image and produce an independent preliminary diagnosis using only visible evidence and the provided diagnostic guideline. Do not mention other experts.

[Guideline]
{guideline}

[Candidate Labels]
{labels}

[Output JSON]
{{
  "features": ["..."],
  "reason": "concise evidence-based reasoning",
  "diagnosis": "one candidate label",
  "confidence": 0.0
}}
"""


DISAGREEMENT_ANALYST_PROMPT = """[Role]
You are a Disagreement Analyst. You do not make the final diagnosis.

[Task]
Audit two preliminary reports and identify why they conflict. Explicitly separate:
1. visual_evidence_disagreement: disagreement about visible image findings;
2. logical_reasoning_disagreement: different diagnostic attribution based on similar findings.

[Expert A]
{report_a}

[Expert B]
{report_b}

[Guideline]
{guideline}

[Output JSON]
{{
  "disagreement_type": "visual|logical|mixed|none",
  "visual_evidence_disagreement": "...",
  "logical_reasoning_disagreement": "...",
  "flaws_of_a": ["..."],
  "flaws_of_b": ["..."],
  "reliable_evidence": ["..."],
  "audit_summary": "..."
}}
"""


ADJUDICATOR_PROMPT = """[Role]
You are a senior adjudicator for difficult medical imaging cases.

[Task]
Make the final diagnosis by synthesizing the raw image, preliminary reports, analyst flaw report, and expert-reviewed diagnostic guideline. Focus on verified visual evidence and correct logical flaws.

[Expert A]
{report_a}

[Expert B]
{report_b}

[Analyst Flaw Report]
{flaw_report}

[Guideline]
{guideline}

[Candidate Labels]
{labels}

[Output JSON]
{{
  "verified_evidence": ["..."],
  "rejected_evidence": ["..."],
  "decision_rationale": "...",
  "diagnosis": "one candidate label",
  "confidence": 0.0
}}
"""


@dataclass(frozen=True)
class PromptBundle:
    """Prompt templates for a single dataset/task."""

    cot: str = TWO_STAGE_COT_TEMPLATE
    stage1: str = EXPERT_DIAGNOSIS_PROMPT
    stage2: str = DISAGREEMENT_ANALYST_PROMPT
    stage3: str = ADJUDICATOR_PROMPT
    source_dir: Optional[str] = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _canonical_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


class PromptBank:
    """Load dataset-specific CoT and MEC prompts from your prompt directory.

    The loader is intentionally flexible: it searches both exact and canonicalized
    dataset directories and accepts filename patterns such as ``*_stage1.txt`` or
    ``br_stage1.txt``. If a file is missing, the corresponding built-in fallback
    prompt is used.
    """

    def __init__(self, prompt_root: Optional[str] = None) -> None:
        self.prompt_root = Path(prompt_root).expanduser() if prompt_root else None

    def _find_dataset_dir(self, dataset: str) -> Optional[Path]:
        if self.prompt_root is None or not self.prompt_root.exists():
            return None
        exact = self.prompt_root / dataset
        if exact.exists():
            return exact
        target = _canonical_name(dataset)
        for p in self.prompt_root.iterdir():
            if p.is_dir() and _canonical_name(p.name) == target:
                return p
        return None

    @staticmethod
    def _first_match(root: Path, patterns: Sequence[str]) -> Optional[Path]:
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
        return None

    def load(self, dataset: str) -> PromptBundle:
        ds_dir = self._find_dataset_dir(dataset)
        if ds_dir is None:
            return PromptBundle()

        mec_dir = ds_dir / "MEC"
        cot_file = self._first_match(ds_dir, ["*cot*.txt", "*CoT*.txt", "cot.txt"])
        stage1_file = self._first_match(mec_dir, ["*stage1*.txt", "*Stage1*.txt", "stage1.txt"]) if mec_dir.exists() else None
        stage2_file = self._first_match(mec_dir, ["*stage2*.txt", "*Stage2*.txt", "stage2.txt"]) if mec_dir.exists() else None
        stage3_file = self._first_match(mec_dir, ["*stage3*.txt", "*Stage3*.txt", "stage3.txt"]) if mec_dir.exists() else None

        return PromptBundle(
            cot=_read_text(cot_file) if cot_file else TWO_STAGE_COT_TEMPLATE,
            stage1=_read_text(stage1_file) if stage1_file else EXPERT_DIAGNOSIS_PROMPT,
            stage2=_read_text(stage2_file) if stage2_file else DISAGREEMENT_ANALYST_PROMPT,
            stage3=_read_text(stage3_file) if stage3_file else ADJUDICATOR_PROMPT,
            source_dir=str(ds_dir),
        )


def render_prompt(template: str, context: Dict[str, object], fallback_context_title: str = "Runtime Context") -> str:
    """Render a prompt while preserving user-written templates.

    Many manually written prompts contain literal braces or custom placeholders.
    Instead of using ``str.format`` directly, this function replaces only common
    placeholders and appends any remaining runtime context in a structured block.
    """
    rendered = template
    aliases = {
        "MEDICAL_CASE": "image_path",
        "IMAGE_PATH": "image_path",
        "LABELS": "labels",
        "CANDIDATE_LABELS": "labels",
        "GUIDELINE": "guideline",
        "GUIDELINES": "guideline",
        "REPORT_A": "report_a",
        "REPORT_B": "report_b",
        "EXPERT_A": "report_a",
        "EXPERT_B": "report_b",
        "FLAW_REPORT": "flaw_report",
        "ANALYST_REPORT": "flaw_report",
        "BASE_PREDICTION": "base_prediction",
        "BASE_CONFIDENCE": "base_confidence",
    }
    for placeholder, key in aliases.items():
        value = context.get(key)
        if value is not None:
            rendered = rendered.replace("{" + placeholder + "}", str(value))
            rendered = rendered.replace("<" + placeholder + ">", str(value))

    appendix_lines = []
    for key in [
        "image_path", "labels", "base_prediction", "base_confidence",
        "guideline", "report_a", "report_b", "flaw_report",
    ]:
        value = context.get(key)
        if value not in (None, ""):
            appendix_lines.append(f"[{key}]\n{value}")
    if appendix_lines:
        rendered = rendered.rstrip() + f"\n\n[{fallback_context_title}]\n" + "\n\n".join(appendix_lines)
    return rendered
