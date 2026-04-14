# Final Project Deliverable Guide

This document helps you complete the final report and technical submission requested in `README.md`.

## 1. Customer context (real-estate marketplace)

Include:

- Business problem to solve with image classification.
- Target users (buyers, sellers, agents, platform operators).
- Expected value (faster listing quality checks, improved search/filtering, better user trust).
- Operational constraints (latency, cost, model update cadence, privacy/security).

## 2. System architecture

Describe the architecture with components and data flow:

- Dataset ingestion and train/validation split.
- Transfer-learning training pipeline.
- Model artifact storage.
- FastAPI inference backend.
- Streamlit frontend.
- W&B experiment tracking.

Suggested diagram nodes:

- `Dataset -> Training (PyTorch) -> Checkpoint -> FastAPI -> Streamlit`
- `Training -> W&B Runs`

## 3. Modeling approach

Document:

- Selected pre-trained model: ResNet50.
- Fine-tuning strategy (`unfreeze_blocks` value, optimizer, LR, epochs, image size).
- Data augmentation and normalization.
- Final checkpoint used in production API.

Reference implementation:

- `final_project/modeling.py`
- `final_project/train.py`

## 4. Experimentation process (W&B)

Document:

- Hyperparameter search strategy (manual grid or staged tuning).
- Tracked variables (learning rate, unfreeze blocks, epochs, batch size, val accuracy).
- Run comparison and final model selection criteria.

Execution command:

```bash
python -m final_project.train --use-wandb --wandb-project mlii-final-project
```

## 5. Performance metrics per class

Use generated files:

- `final_project/reports/classification_report.json`
- `final_project/reports/confusion_matrix.csv`
- `final_project/reports/summary_metrics.json`

Report:

- Accuracy.
- Per-class precision, recall, F1-score.
- Confusion matrix analysis and key failure modes.
- Business interpretation (what errors are acceptable, what errors are costly).

## 6. API documentation

Backend source:

- `final_project/api/main.py`

Run and validate:

```bash
uvicorn final_project.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger:

- `http://127.0.0.1:8000/docs`

Document endpoint behavior:

- `GET /health`
- `GET /metadata`
- `POST /predict`

## 7. Deployment artifacts

Frontend source:

- `final_project/streamlit_app.py`

Run:

```bash
streamlit run final_project/streamlit_app.py --server.address 127.0.0.1 --server.port 8601
```

## 8. Required project links

Fill these placeholders in your report:

- Git repository (public): `<PASTE_GIT_URL>`
- W&B project/workspace: `<PASTE_WANDB_URL>`

## 9. Access requirements

Invite these users to your W&B workspace:

- `agascon@comillas.edu`
- `rkramer@comillas.edu`

## 10. Final submission checklist

- Reproducible codebase with setup instructions.
- Working API with accessible Swagger docs.
- Working Streamlit app connected to API.
- Traceable W&B experimentation history.
- Final report (max 6 pages, no cover page) with conclusions and business recommendations.
