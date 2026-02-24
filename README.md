# IT Ticket Triage NLP System

An end-to-end NLP project for automated IT ticket triage.

The system predicts:
- routing department
- urgency/priority
- key tags
- short summary

It is built with a training notebook first, then packaged into a Streamlit app for live demo and product-style usage.

## Objective

Manual ticket triage is slow, inconsistent, and expensive at scale. This project automates first-pass triage using NLP so support teams can:
- reduce time-to-first-action
- reduce misroutes
- standardize triage decisions
- estimate business savings from automation

## Target Architecture (E2E)

```mermaid
flowchart LR
    A[Ticket Intake\nshort description + full description] --> B[Preprocessing\nclean + normalize + combine to ticket_text]
    B --> C1[Department Classifier\nTransformer]
    B --> C2[Urgency Classifier\nTransformer]
    B --> C3[Tags Extractor\nYAKE]
    B --> C4[Summary Generator\nT5-small]
    C1 --> D[Unified JSON Output]
    C2 --> D
    C3 --> D
    C4 --> D
    D --> E[Demo UI\nStreamlit]
    D --> F[Business Value Analysis]
```

## JSON Output Schema

```json
{
  "ticket_id": "...",
  "department": {"label": "...", "confidence": 0.92},
  "urgency": {"label": "...", "confidence": 0.81},
  "tags": ["vpn", "login", "timeout"],
  "summary": "User cannot connect to VPN after password reset."
}
```

## What Was Implemented

### 1) Data Pipeline
- Raw data ingestion from `data/raw/IT Support Ticket Data.csv`
- Stratified subset generation script:
  - `scripts/create_stratified_subset.py`
  - default output: `data/processed/IT Support Ticket Data.stratified_3000.csv`
- Stratified train/val/test split with leakage guard by ticket id

### 2) Text Preprocessing
- HTML stripping
- URL/email removal
- whitespace normalization
- conservative boilerplate cleanup (headers/signoff/disclaimer handling)
- stable `ticket_text` field used across all tasks

### 3) Label Standardization
- Department alias normalization (example: `billing & payments` -> `Billing and Payments`)
- Urgency alias normalization (example: `high priority` -> `high`)
- Mapping persistence (`label2id`, `id2label`) for reproducible inference

### 4) Multi-Approach Modeling
- Baseline: TF-IDF + Logistic Regression (department + urgency)
- Transformer classifiers with Hugging Face `Trainer`
- Candidate comparison framework for model and learning-rate trials
- Early stopping support
- Final model export under:
  - `models/department_model/best`
  - `models/urgency_model/best`

### 5) Tags + Summary Module
- Tags: YAKE keyphrase extraction (fast and robust)
- Summary: `t5-small` abstractive summary (optional toggle in app)

### 6) Unified Inference + Demo App
- Shared runtime engine: `app/triage_engine.py`
- Streamlit UI: `app/main.py`
- Single-call inference: `triage_ticket(...)`

## What Was Tested / Compared

- Baseline vs Transformer for both tasks (department + urgency)
- Candidate trials with configurable learning rates and epochs
- Inference smoke tests using saved model checkpoints
- End-to-end demo path: input ticket -> JSON output

Evaluation artifacts are written under `results/` and include metrics, predictions, and model-selection outputs.

## Repository Layout

```text
NLP-IT-Ticket_Triage/
  .streamlit/
    config.toml              # Streamlit theme and server settings
  app/
    main.py                  # Streamlit UI entry point (tabs, sidebar, styles)
    triage_engine.py         # Shared inference runtime (classifiers, tags, summary)
  data/
    raw/                     # Original unprocessed CSV dataset
    processed/               # Stratified/cleaned splits used for training
  docs/                      # Supplementary documentation and notes
  models/
    department_model/        # Trained department classifier (Git LFS)
    urgency_model/           # Trained urgency classifier (Git LFS)
  notebooks/
    01_training_and_eval.ipynb  # End-to-end training, evaluation, and analysis
  results/                   # Evaluation metrics, predictions, model-selection logs
  scripts/
    create_stratified_subset.py  # Generates balanced subset from raw data
  requirements.txt           # Python dependencies for running the app
```

## Quick Start (No Training Required)

This repository already includes trained inference checkpoints in `models/*/best` tracked with Git LFS.
You can run the app directly without training from scratch.

### 1) Install Git LFS (one-time on your machine)

```bash
git lfs install
```

### 2) Clone and download model binaries

```bash
git clone https://github.com/rawad-yared/NLP-IT-Ticket_Triage.git
cd NLP-IT-Ticket_Triage
git lfs pull
```

### 3) Create virtual environment + install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4) Run Streamlit

```bash
python -m streamlit run app/main.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`).

Required model folders for app inference:
- `models/department_model/best`
- `models/urgency_model/best`

## Optional: Train Locally (If You Want to Rebuild Models)

### 1) (Optional) Regenerate stratified subset

```bash
python scripts/create_stratified_subset.py
```

### 2) Train/evaluate in notebook

```bash
jupyter notebook notebooks/01_training_and_eval.ipynb
```

Run cells top-to-bottom to produce updated checkpoints and metrics.

### 3) Re-run app with your new checkpoints

```bash
python -m streamlit run app/main.py
```

## Notes on First Run Downloads

- If you pulled models via Git LFS, department/urgency checkpoints are already local.
- `t5-small` summary model may still download on first use if summary is enabled.
- If summary is disabled in the app, no summary-model download is needed.

## Common Troubleshooting

### `ModuleNotFoundError: No module named 'yake'`

Usually means Streamlit is running outside your project venv.

Use:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/main.py
```

### Streamlit opens but wrong environment is used

Always launch with module form:

```bash
python -m streamlit run app/main.py
```

This guarantees the same interpreter as your active venv.

### App cannot find model weights after clone

You likely cloned pointers without downloading LFS binaries. Run:

```bash
git lfs pull
```

## Current Status

- Notebook pipeline is implemented and runnable locally.
- Streamlit app is implemented and connected to trained model outputs.
- Project is ready for iterative tuning and final presentation packaging.
