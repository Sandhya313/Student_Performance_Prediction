import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sklearn.impute import SimpleImputer

# Resolve base directory and model path relative to main.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

EXPECTED_FEATURES: List[str] = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score",
]

# Global model state
model_pipeline: Any = None
model_load_error: Optional[str] = None


def _ensure_compatibility(estimator: Any) -> None:
    """Ensure cross-version scikit-learn unpickling compatibility (e.g., SimpleImputer attributes)."""
    if hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            _ensure_compatibility(step)
    elif hasattr(estimator, "transformers_"):
        for _, trans, _ in estimator.transformers_:
            _ensure_compatibility(trans)
    elif hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            _ensure_compatibility(step)
    if isinstance(estimator, SimpleImputer):
        if not hasattr(estimator, "_fill_dtype") and hasattr(estimator, "_fit_dtype"):
            estimator._fill_dtype = estimator._fit_dtype


def load_model() -> None:
    """Load model.joblib from the path relative to main.py."""
    global model_pipeline, model_load_error
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        loaded = joblib.load(MODEL_PATH)
        _ensure_compatibility(loaded)
        model_pipeline = loaded
        model_load_error = None
    except Exception as e:
        model_pipeline = None
        model_load_error = str(e)


# Load at module import so tests and runners have immediate access
load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler."""
    load_model()
    yield


app = FastAPI(
    title="Student Placement Prediction API",
    description=(
        "Production-ready FastAPI service for predicting student placement status "
        "using a pre-trained scikit-learn pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


class StudentFeatures(BaseModel):
    """Pydantic model validating the 6 input features for student placement prediction."""
    model_config = ConfigDict(extra="forbid")

    study_hours: float = Field(
        ...,
        description="Daily study hours",
        ge=0.0,
        examples=[5.0],
    )
    attendance: float = Field(
        ...,
        description="Attendance percentage (0-100)",
        ge=0.0,
        le=100.0,
        examples=[85.0],
    )
    sleep_hours: float = Field(
        ...,
        description="Daily sleep hours",
        ge=0.0,
        examples=[7.0],
    )
    internet_usage: float = Field(
        ...,
        description="Daily internet usage in hours",
        ge=0.0,
        examples=[3.0],
    )
    assignments_completed: float = Field(
        ...,
        description="Total assignments completed",
        ge=0.0,
        examples=[8.0],
    )
    previous_score: float = Field(
        ...,
        description="Previous exam or aggregate score (0-100)",
        ge=0.0,
        le=100.0,
        examples=[75.0],
    )


class PredictionResponse(BaseModel):
    placement_status: str
    probabilities: Optional[Dict[str, float]] = None
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    expected_features: List[str]
    error: Optional[str] = None


@app.get("/", summary="API Welcome Message")
def read_root():
    """Welcome endpoint with service information and quick links."""
    return {
        "message": "Welcome to the Student Placement Prediction API",
        "docs_url": "/docs",
        "health_url": "/health",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Health check endpoint reporting service and model loading status."""
    is_loaded = model_pipeline is not None
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        expected_features=EXPECTED_FEATURES,
        error=model_load_error,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Placement Status")
def predict_placement(payload: StudentFeatures):
    """
    Predict student placement status based on six academic and behavioural features.
    Rejects any extraneous fields (e.g. exam_score) to prevent target leakage.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded or failed to initialize: {model_load_error}",
        )

    try:
        # Build pandas DataFrame with exact feature names and order expected by the pipeline
        data_dict = {
            "study_hours": [payload.study_hours],
            "attendance": [payload.attendance],
            "sleep_hours": [payload.sleep_hours],
            "internet_usage": [payload.internet_usage],
            "assignments_completed": [payload.assignments_completed],
            "previous_score": [payload.previous_score],
        }
        input_df = pd.DataFrame(data_dict)

        # Predict placement status
        raw_pred = model_pipeline.predict(input_df)
        placement_status = str(raw_pred[0])

        # Extract prediction probabilities if supported by the model
        probabilities: Optional[Dict[str, float]] = None
        if hasattr(model_pipeline, "predict_proba"):
            proba_matrix = model_pipeline.predict_proba(input_df)
            if proba_matrix is not None and len(proba_matrix) > 0:
                proba_row = proba_matrix[0]
                classes = getattr(
                    model_pipeline,
                    "classes_",
                    [f"class_{i}" for i in range(len(proba_row))],
                )
                probabilities = {
                    str(cls_name): round(float(prob), 4)
                    for cls_name, prob in zip(classes, proba_row)
                }

        return PredictionResponse(
            placement_status=placement_status,
            probabilities=probabilities,
            model_version="1.0.0",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction computation failed: {str(e)}",
        )
