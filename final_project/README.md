# Final Project Pipeline

This folder provides an end-to-end pipeline for the final deliverable:

- Transfer-learning training on the scene dataset.
- Class-wise evaluation with precision/recall/F1 and confusion matrix.
- FastAPI inference service with Swagger docs.
- Streamlit frontend connected to the API.
- Optional Weights & Biases tracking.

## 1) Install dependencies

From the repository root:

```bash
pip install -r final_project/requirements.txt
```

## 2) Train the model

Default dataset paths are already connected to:

- `03TransferLearning/dataset/training`
- `03TransferLearning/dataset/validation`

Run baseline training:

```bash
python -m final_project.train \
  --epochs 6 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --unfreeze-blocks 1
```

Artifacts are saved by default at:

- `final_project/artifacts/resnet50_scene_best.pt`
- `final_project/artifacts/resnet50_scene_best.history.csv`
- `final_project/artifacts/resnet50_scene_best.summary.json`

### Optional: Enable W&B logging

Set your key in the current shell before training:

```bash
# PowerShell
$env:WANDB_API_KEY = "<YOUR_KEY>"

python -m final_project.train --use-wandb --wandb-project mlii-final-project
```

## 3) Evaluate class-wise metrics

```bash
python -m final_project.evaluate \
  --checkpoint final_project/artifacts/resnet50_scene_best.pt
```

Outputs are saved in:

- `final_project/reports/classification_report.json`
- `final_project/reports/confusion_matrix.csv`
- `final_project/reports/summary_metrics.json`

## 4) Run FastAPI backend

```bash
uvicorn final_project.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger/OpenAPI docs:

- http://127.0.0.1:8000/docs

Useful environment variables:

- `FINAL_PROJECT_CHECKPOINT` (custom checkpoint path)
- `FINAL_PROJECT_CLASS_NAMES_DIR` (directory containing class folders)

## 5) Run Streamlit frontend

```bash
streamlit run final_project/streamlit_app.py --server.address 127.0.0.1 --server.port 8601
```

Environment variable:

- `FINAL_PROJECT_API_URL` (default: `http://127.0.0.1:8000`)

## 6) Deliverable checklist mapping

- Modeling approach: `final_project/train.py`, `final_project/modeling.py`
- Experimentation and W&B: `final_project/train.py` (`--use-wandb`)
- Per-class metrics: `final_project/evaluate.py` and reports folder
- API docs and behavior: FastAPI app and `/docs`
- Production frontend: `final_project/streamlit_app.py`
