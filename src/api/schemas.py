"""
src/api/schemas.py
==================
Pydantic request / response schemas for the Forex prediction API.

Keeping schemas in a dedicated module makes them independently
testable and decouples validation logic from route handlers.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request schemas ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Single-row online inference request.

    The caller passes a list of (date, currency_code) rows; the server
    applies the full feature-engineering + scaling pipeline and returns
    predictions in the same order.

    Fields
    ------
    rows          : List of dicts each containing at least
                    'date' (ISO-8601 string) and 'currency_code' (str).
                    Additional fields (e.g. 'exchange_rate') are forwarded
                    to the feature engineering pipeline when present.
    model         : Which artifact to use for inference.
                    'lgb' | 'xgb' | 'stacking' (default: 'stacking').
    return_uncertainty : When True, run MC-Dropout on DL sub-models and
                         return per-prediction std-dev alongside the mean.
                         Only meaningful when model == 'stacking'.
    """
    rows: List[dict] = Field(
        ...,
        min_length=1,
        description="List of input rows. Each must contain 'date' and 'currency_code'.",
    )
    model: str = Field(
        default="stacking",
        description="Model artifact to use: lgb | xgb | stacking.",
    )
    return_uncertainty: bool = Field(
        default=False,
        description="Return MC-Dropout uncertainty estimates (stacking only).",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = {"lgb", "xgb", "stacking"}
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}, got '{v}'")
        return v

    @field_validator("rows")
    @classmethod
    def validate_rows(cls, v: list) -> list:
        for i, row in enumerate(v):
            if "date" not in row:
                raise ValueError(f"Row {i} missing 'date' field.")
            if "currency_code" not in row:
                raise ValueError(f"Row {i} missing 'currency_code' field.")
        return v


class BatchFileRequest(BaseModel):
    """
    Metadata for file-based batch inference (used alongside UploadFile).
    Validated separately from the file upload form data.
    """
    model: str = Field(default="stacking")
    return_uncertainty: bool = Field(default=False)


# ── Response schemas ───────────────────────────────────────────────────────────

class PredictionRow(BaseModel):
    """A single prediction result row."""
    date: str
    currency_code: str
    predicted_rate: float
    uncertainty: Optional[float] = Field(
        default=None,
        description="MC-Dropout std-dev. Populated only when return_uncertainty=True.",
    )


class PredictResponse(BaseModel):
    """Successful prediction response."""
    model_used: str
    n_predictions: int
    predictions: List[PredictionRow]


class HealthResponse(BaseModel):
    """API health-check response."""
    status: str
    model_dir: str
    artifacts_loaded: bool
    models_available: List[str]


class ModelInfoResponse(BaseModel):
    """Metadata about a loaded model artifact."""
    model_name: str
    model_dir: str
    artifact_files: List[str]
    n_currencies: int
    feature_config_path: str
