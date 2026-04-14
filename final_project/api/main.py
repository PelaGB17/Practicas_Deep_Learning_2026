import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from final_project.data import list_class_names_from_directory
from final_project.inference import ScenePredictor
from final_project.settings import DEFAULT_CHECKPOINT_PATH, DEFAULT_TRAIN_DIR

MODEL_CHECKPOINT_ENV = "FINAL_PROJECT_CHECKPOINT"
CLASS_NAMES_DIR_ENV = "FINAL_PROJECT_CLASS_NAMES_DIR"

app = FastAPI(
    title="Final Project Scene Classifier API",
    description="FastAPI backend for transfer-learning scene image classification.",
    version="1.0.0",
)

# Defaults make health endpoint safe even if startup has not executed yet.
app.state.predictor = None
app.state.error = "Startup hook has not initialized the model yet."
app.state.checkpoint_path = str(DEFAULT_CHECKPOINT_PATH)


def _get_checkpoint_path() -> Path:
    raw_path = os.getenv(MODEL_CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT_PATH))
    return Path(raw_path)


def _get_optional_class_names() -> Optional[list]:
    raw_dir = os.getenv(CLASS_NAMES_DIR_ENV, str(DEFAULT_TRAIN_DIR))
    class_dir = Path(raw_dir)
    if not class_dir.exists():
        return None
    try:
        return list_class_names_from_directory(class_dir)
    except Exception:
        return None


@app.on_event("startup")
def startup_event() -> None:
    app.state.predictor = None
    app.state.error = None
    app.state.checkpoint_path = str(_get_checkpoint_path())

    checkpoint_path = _get_checkpoint_path()
    if not checkpoint_path.exists():
        app.state.error = (
            f"Checkpoint not found at {checkpoint_path}. "
            "Train a model or set FINAL_PROJECT_CHECKPOINT."
        )
        return

    try:
        app.state.predictor = ScenePredictor(
            checkpoint_path=checkpoint_path,
            class_names=_get_optional_class_names(),
        )
    except Exception as exc:
        app.state.error = str(exc)
        app.state.predictor = None


@app.get("/health")
def health() -> dict:
    predictor = getattr(app.state, "predictor", None)
    error = getattr(app.state, "error", None)
    checkpoint_path = getattr(app.state, "checkpoint_path", str(_get_checkpoint_path()))
    loaded = predictor is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "checkpoint_path": checkpoint_path,
        "detail": error,
    }


@app.get("/metadata")
def metadata() -> dict:
    predictor = getattr(app.state, "predictor", None)
    error = getattr(app.state, "error", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail=error or "Model is not loaded.")

    return {
        "num_classes": len(predictor.class_names),
        "class_names": predictor.class_names,
        "device": predictor.device,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    predictor = getattr(app.state, "predictor", None)
    error = getattr(app.state, "error", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail=error or "Model is not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predictor.predict_bytes(contents, top_k=3)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    return {
        "filename": file.filename,
        "status": "success",
        **result,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("final_project.api.main:app", host="127.0.0.1", port=8000, reload=True)
