"""Debug-only MEC demo with dataset-specific prompt loading.

Replace EchoAgent with your actual MedGemma/GPT/Gemini clients during real inference.
"""

import torch

from himec.mec import EchoAgent, GuidelineStore, MultiExpertConsultation, UncertaintyRouter

labels = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
base_logits = torch.tensor([0.1, 0.1, 0.2, 0.1, 0.1, 0.1])

mec = MultiExpertConsultation(
    expert_a=EchoAgent("ACK"),
    expert_b=EchoAgent("MEL"),
    analyst=EchoAgent(),
    adjudicator=EchoAgent("ACK"),
    guideline_store=GuidelineStore(guideline_root="guidelines"),
    router=UncertaintyRouter(threshold=0.75),
    prompt_root="prompts",
    dataset="PAD-UFES-20",
    trace_path="runs/himec/mec_demo_trace.jsonl",
)
print(mec.consult("case.png", base_logits, labels, force=True).to_json())
