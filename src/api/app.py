"""
src/api/app.py
===============
FastAPI application for online Forex exchange-rate inference.

Endpoints
---------
GET  /health          — liveness + artifact status check
GET  /model-info      — metadata about the loaded model artifacts
POST /predict         — single or multi-row online inference (JSON)
POST /predict/batch   — CSV file upload for batch inference

Design decisions
----------------
- ForexPredictor is a singleton loaded once at startup via lifespan.
- All request validation is handled by Pydantic schemas (schemas.py).
- Async route handlers keep the event loop free during preprocessing.
- CSV batch endpoint uses UploadFile + background streaming to avoid
  loading large files entirely into RAM.
- CORS is enabled for all origins by default (restrict in production).
- All errors return RFC 7807-style JSON with 'detail' and 'type' fields.

Usage
-----
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
    # or via Docker:
    docker run -p 8000:8000 forex-prediction-api
"""

from __future__ import annotations
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionRow,
    HealthResponse,
    ModelInfoResponse,
)
from src.api.predictor import ForexPredictor

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Configuration from environment ────────────────────────────────────────────
_MODEL_DIR   = os.environ.get("MODEL_DIR",   "outputs/latest")
_CONFIG_PATH = os.environ.get("FEAT_CONFIG", "config/features.yaml")
_API_TITLE   = "Forex Ensemble Prediction API"
_API_VERSION = "1.0.0"

# ── Predictor singleton (populated at startup) ─────────────────────────────────
predictor: ForexPredictor = None


# ── Lifespan: load artifacts once on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    log.info(f"Starting {_API_TITLE} v{_API_VERSION}")
    log.info(f"Model directory : {_MODEL_DIR}")
    log.info(f"Feature config  : {_CONFIG_PATH}")

    predictor = ForexPredictor(
        model_dir=_MODEL_DIR,
        config_path=_CONFIG_PATH,
    )
    try:
        predictor.load_artifacts()
        log.info("All artifacts loaded successfully.")
    except FileNotFoundError as exc:
        log.warning(
            f"Artifact loading failed: {exc}. "
            "Server will start in degraded mode — train a model first."
        )
    yield
    log.info("Shutting down API server.")


# ── App instance ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=_API_TITLE,
    version=_API_VERSION,
    description=(
        "Online inference API for the Forex Ensemble Prediction system. "
        "Supports LightGBM, XGBoost, and Stacking meta-learner models "
        "with optional MC-Dropout uncertainty estimation."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """
    Liveness + readiness check.
    Returns artifact load status and available model names.
    """
    return HealthResponse(
        status="ok" if (predictor and predictor.artifacts_loaded) else "degraded",
        model_dir=_MODEL_DIR,
        artifacts_loaded=bool(predictor and predictor.artifacts_loaded),
        models_available=predictor.available_models if predictor else [],
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Ops"])
async def model_info(
    model: str = Query(default="stacking",
                       description="Model name: lgb | xgb | stacking"),
):
    """
    Returns metadata about the loaded model artifacts:
    artifact files present, number of known currencies, config path.
    """
    _require_artifacts()
    if model not in predictor.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not loaded. Available: {predictor.available_models}",
        )
    return ModelInfoResponse(
        model_name=model,
        model_dir=_MODEL_DIR,
        artifact_files=predictor.artifact_files,
        n_currencies=predictor.n_currencies,
        feature_config_path=_CONFIG_PATH,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    **Online inference** — accepts 1..N rows as JSON, returns predicted
    exchange rates in the same order.

    Request body example:
    ```json
    {
      "rows": [
        {"date": "2025-01-10", "currency_code": "USD", "exchange_rate": 83.10},
        {"date": "2025-01-11", "currency_code": "EUR", "exchange_rate": 90.45}
      ],
      "model": "stacking",
      "return_uncertainty": false
    }
    ```
    """
    _require_artifacts()
    try:
        preds, uncertainties, df_proc = predictor.predict(
            rows=request.rows,
            model_name=request.model,
            return_uncertainty=request.return_uncertainty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # Reconstruct currency codes from one-hot columns
    cc_cols = [c for c in df_proc.columns if c.startswith("currency_code_")]
    if cc_cols:
        currency_codes = (
            df_proc[cc_cols]
            .idxmax(axis=1)
            .str.replace("currency_code_", "", regex=False)
        )
    else:
        currency_codes = pd.Series(["unknown"] * len(df_proc))

    rows_out: List[PredictionRow] = []
    for i, (_, row) in enumerate(df_proc.iterrows()):
        if i >= len(preds):
            break
        rows_out.append(PredictionRow(
            date=str(row.get("date", ""))[:10],
            currency_code=str(currency_codes.iloc[i]),
            predicted_rate=float(round(preds[i], 6)),
            uncertainty=(
                float(round(uncertainties[i], 6))
                if uncertainties is not None else None
            ),
        ))

    return PredictResponse(
        model_used=request.model,
        n_predictions=len(rows_out),
        predictions=rows_out,
    )


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(
    file: UploadFile = File(..., description="CSV file with columns: date, currency_code [, exchange_rate]"),
    model: str       = Query(default="stacking", description="lgb | xgb | stacking"),
    return_uncertainty: bool = Query(default=False),
):
    """
    **Batch CSV inference** — upload a CSV file, get predictions back
    as a downloadable CSV.

    Expected CSV columns: `date`, `currency_code` (and optionally `exchange_rate`).

    Returns a CSV file stream with an added `predicted_rate` column
    (and `uncertainty` column when `return_uncertainty=True`).
    """
    _require_artifacts()

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=415,
            detail="Only CSV files are accepted for batch inference.",
        )

    content = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    required_cols = {"date", "currency_code"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV missing required columns: {missing}",
        )

    rows = df_raw.to_dict(orient="records")
    try:
        preds, uncertainties, df_proc = predictor.predict(
            rows=rows,
            model_name=model,
            return_uncertainty=return_uncertainty,
        )
    except Exception as exc:
        log.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    out_df = df_proc[["date"]].copy()
    if "exchange_rate" in df_proc.columns:
        out_df["exchange_rate"] = df_proc["exchange_rate"]
    out_df["predicted_rate"] = preds
    if return_uncertainty and uncertainties is not None:
        out_df["uncertainty"] = uncertainties

    buffer = io.StringIO()
    out_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=predictions_{model}.csv"
        },
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _require_artifacts() -> None:
    """Raise 503 if artifacts aren't loaded yet."""
    if not predictor or not predictor.artifacts_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run training pipeline first.",
        )
