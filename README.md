# CausalCrash Scripts (Anonymous)

This repository contains inference, judge, and reporting scripts used for the
CausalCrash evaluation pipeline. It is intended for NeurIPS E&D review and does
not include dataset files or model weights.

- Dataset: https://huggingface.co/datasets/meet2008/CausalCrash
- Paper: NeurIPS 2026 Evaluations & Datasets Track Submission #3454

## Contents

- scripts/
  - openrouter_infer.py
  - judge.py (GPT-4.1-mini)
  - judge2.py (Gemini 2.5 Flash)
  - report.py
  - model folders with gen_answers.py and serve_model.sh
- requirements.txt
- install_deps.sh
- .env.example

## Setup

1) Create environment

```bash
conda create -n causalcrash -y python==3.12.* nvidia::cuda-toolkit==13.0.*
conda activate causalcrash
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

2) Set environment variables

Copy .env.example to .env and fill in values, or export variables directly.

## Data

Dataset is hosted on Hugging Face:
https://huggingface.co/datasets/meet2008/CausalCrash

This repo expects a local dataset directory such as:

```
__data/
  dataset/           # videos (downloaded locally)
  annotations/       # ground-truth JSONs
```

Notes:
- The dataset provides metadata and annotations. Raw videos are not hosted; use
  `videos.csv` URLs to fetch videos and place them in `__data/dataset/`.
- Ground-truth annotations should be placed in `__data/annotations/` with
  per-video JSONs (or convert from the provided `annotations.json`).

## End-to-End Workflow

1) Run model inference (generate predicted JSONs)
2) Load ground-truth JSONs and run judges
3) Generate aggregate report from both judges

## Step 1: Inference

### OpenRouter-based inference

```bash
python scripts/openrouter_infer.py
```

### Local vLLM inference

Each model folder contains:
- serve_model.sh
- gen_answers.py

Example:

```bash
bash scripts/InternVL3-14B/serve_model.sh
python scripts/InternVL3-14B/gen_answers.py
```

## Step 2: LLM-as-a-judge

Set these environment variables (see `.env.example`):

- `MODEL_TAG`: model name for labeling outputs
- `PRED_DIR`: directory with predicted JSONs
- `GT_DIR`: directory with ground-truth JSONs
- `OUT_DIR`: output directory for judge results

Example layout:

```
outputs/
  model_answers/<MODEL_TAG>/   # predictions
  judge_results/gpt-4.1-mini/<MODEL_TAG>/
  judge_results/gemini-2.5-flash/<MODEL_TAG>/
```

```bash
python scripts/judge.py
python scripts/judge2.py
```

Outputs are written to `OUT_DIR` (see environment variables).

## Step 3: Report generation

Set:

- `GEMINI_DIR`: directory with judge2 outputs
- `GPT_DIR`: directory with judge outputs
- `REPORT_OUT_DIR`: where plots/summary will be written

```bash
python scripts/report.py
```

## Expected Output
After running report.py, you should see:
- Per-model scores across L1, L2, L4, L5 reasoning levels
- Weighted CausalCrash Score (CCS) per model
- Inter-judge disagreement and bias plots
- Pearson correlation between GPT-4.1-mini and Gemini 2.5 Flash judges

## Notes

- API keys are loaded from environment variables. Do not hard-code keys.
- Output directories are configurable via environment variables.
- This repo is anonymous and excludes private data and weights.
