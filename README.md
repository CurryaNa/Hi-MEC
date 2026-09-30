# Hi-MEC reference implementation

This package is a paper-aligned implementation of **Hi-MEC: Hierarchical Medical Diagnosis via Quality-Aware Knowledge Fusion and Multi-Expert Consultation**.

Main components:

- **Two-stage CoT knowledge generation**: objective visual observation followed by attribute-level description.
- **QACDF**: frozen **BioMedCLIP** estimates attribute-level visual reliability; BioClinical-BERT learns task-specific text features.
- **Reciprocal Semantic-Visual Alignment**: bidirectional image-to-text and text-to-image cross-attention.
- **EAF**: entropy-gated adaptive fusion with branch-wise auxiliary losses.
- **MEC**: uncertainty-routed multi-expert consultation with Stage 1 preliminary diagnosis, Stage 2 disagreement attribution, and Stage 3 adjudication.

## Recommended project layout

```text
project_root/
  dataset/
    breastmnist/
      image/train/<class>/*.png
      image/val/<class>/*.png
      image/test/<class>/*.png
      text/train/<class>/*.txt
      text/val/<class>/*.txt
      text/test/<class>/*.txt
    DermaMNIST/
    PAD-UFES-20/
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
    PAD-UFES-20/
      pad_cot.txt
      MEC/pad_stage1.txt
      MEC/pad_stage2.txt
      MEC/pad_stage3.txt
  guidelines/
  himec_topconf_code/
```

The code reads your existing MEC prompt files via `--prompt_root` and `--dataset_name`. Missing prompt files automatically fall back to built-in templates.

## Train Hi-MEC Base

```bash
cd himec_topconf_code
python scripts/train_himec.py \
  --data_root ../dataset/breastmnist \
  --num_classes 2 \
  --text_encoder /path/to/Bio_ClinicalBERT \
  --medmamba_checkpoint /path/to/medmamba_ckpt.pth \
  --biomedclip_dir checkpoints/biomedclip \
  --epochs 40 \
  --batch_size 16 \
  --save_dir runs/breastmnist
```

Fast code-path debug without MedMamba/BioMedCLIP:

```bash
python scripts/train_himec.py \
  --data_root ../dataset/breastmnist \
  --num_classes 2 \
  --debug_tiny_visual \
  --disable_biomedclip \
  --epochs 1
```

## Test Hi-MEC Base

```bash
python scripts/test_himec.py \
  --data_root ../dataset/breastmnist \
  --split test \
  --checkpoint runs/breastmnist/best.pt \
  --num_classes 2 \
  --text_encoder /path/to/Bio_ClinicalBERT \
  --biomedclip_dir checkpoints/biomedclip \
  --save_jsonl runs/breastmnist/test_predictions.jsonl
```

The output JSONL contains base logits, probabilities, QACDF attribute weights, EAF modality weights, and entropy values for visualization/analysis.

## Test the MEC pipeline with your prompt files

This reference package does not include external API credentials. Use `--mock_mec` to verify that MEC correctly loads your dataset-specific prompts and executes the full three-stage route:

```bash
python scripts/test_himec.py \
  --data_root ../dataset/PAD-UFES-20 \
  --split test \
  --checkpoint runs/pad/best.pt \
  --num_classes 6 \
  --text_encoder /path/to/Bio_ClinicalBERT \
  --prompt_root ../prompts \
  --dataset_name PAD-UFES-20 \
  --with_mec \
  --mock_mec \
  --mec_threshold 0.75 \
  --save_jsonl runs/pad/test_full_mock.jsonl
```

For real inference, `scripts/test_himec.py --with_mec` now wires the MEC stages as follows:

- Stage 1: one local MedGemma model is sampled twice at high temperature to produce two independent preliminary reports.
- Stage 2: GPT analyzes the disagreement between the two MedGemma reports.
- Stage 3: Gemini adjudicates the final diagnosis with the image, reports, flaw analysis, and guideline.

Set the API key through an environment variable instead of hard-coding it:

```bash
export MEC_API_KEY=your_api_key
export MEC_BASE_URL=https://your-openai-compatible-endpoint/v1/
python scripts/test_himec.py \
  --data_root ../dataset/PAD-UFES-20 \
  --split test \
  --checkpoint runs/pad/best.pt \
  --num_classes 6 \
  --text_encoder /path/to/Bio_ClinicalBERT \
  --prompt_root ../prompts \
  --dataset_name PAD-UFES-20 \
  --with_mec \
  --mec_threshold 0.75 \
  --mec_expert_temperature 1.0 \
  --medgemma_model google/medgemma-1.5-4b-it \
  --gpt_model gpt-5.1 \
  --gemini_model gemini-3-pro-preview \
  --mec_base_url "$MEC_BASE_URL" \
  --save_jsonl runs/pad/test_full_real.jsonl
```

The real clients still follow the original agent signature:

```python
def agent(prompt: str, image_path: str | None = None, temperature: float = 0.2) -> str:
    return '{"features": ["..."], "reason": "...", "diagnosis": "ACK", "confidence": 0.82}'
```

## Dataset notes

Each text file should contain the five attribute-level descriptions produced by your CoT prompt, one per line. Example:

```text
1. Primary Colors: ...
2. Border Appearance: ...
3. Shape Symmetry: ...
4. Surface Texture: ...
5. Structural Features: ...
```

