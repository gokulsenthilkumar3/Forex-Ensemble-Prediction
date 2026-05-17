"""
src/api/app.py
===============
FastAPI application for online Forex exchange-rate inference.

Fixed bugs (see review comment on PR #7):
  - Bug 2: Currency code reconstruction now reads '_currency_code_orig' from
           df_proc (set by predictor._preprocess) instead of using idxmax()
           on OHE columns, which was wrong for the drop_first=True dropped currency.
  - Bug 5: `predictor` global is now typed Optional[ForexPredictor] and
           _require_artifacts() guards against None explicitly.

Endpoints
---------
GET  /health          — liveness + artifact status
GET  /model-info      — artifact metadata
POST /predict         — online JSON inference
POST /predict/batch   — CSV file upload → CSV download
"""

from __future__ import annotations
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

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

_MODEL_DIR   = os.environ.get("MODEL_DIR",   "outputs/latest")
_CONFIG_PATH = os.environ.get("FEAT_CONFIG", "config/features.yaml")
_API_TITLE   = "Forex Ensemble Prediction API"
_API_VERSION = "1.0.0"

# FIX Bug 5: typed as Optional so type-checkers and _require_artifacts() handle None safely.
predictor: Optional[ForexPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    log.info(f"Starting {_API_TITLE} v{_API_VERSION}")
    predictor = ForexPredictor(model_dir=_MODEL_DIR, config_path=_CONFIG_PATH)
    try:
        predictor.load_artifacts()
        log.info("All artifacts loaded successfully.")
    except FileNotFoundError as exc:
        log.warning(
            f"Artifact loading failed: {exc}. "
            "Server starting in degraded mode — run training first."
        )
    yield
    log.info("Shutting down API server.")


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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    return HealthResponse(
        status="ok" if (predictor and predictor.artifacts_loaded) else "degraded",
        model_dir=_MODEL_DIR,
        artifacts_loaded=bool(predictor and predictor.artifacts_loaded),
        models_available=predictor.available_models if predictor else [],
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Ops"])
async def model_info(
    model: str = Query(default="stacking", description="lgb | xgb | stacking"),
):
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
    Online inference — 1..N rows as JSON, predictions returned in the same order.
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

    rows_out: List[PredictionRow] = []
    for i, (_, row) in enumerate(df_proc.iterrows()):
        if i >= len(preds):
            break
        # FIX Bug 2: Read from '_currency_code_orig' which is always populated,
        # even for the currency that get_dummies(drop_first=True) dropped.
        cc = str(row.get("_currency_code_orig", "unknown"))
        rows_out.append(PredictionRow(
            date=str(row.get("date", ""))[:10],
            currency_code=cc,
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
    file: UploadFile = File(..., description="CSV with columns: date, currency_code [, exchange_rate]"),
    model: str       = Query(default="stacking", description="lgb | xgb | stacking"),
    return_uncertainty: bool = Query(default=False),
):
    """
    Batch CSV inference — upload a CSV, get predictions back as a downloadable CSV.
    Output CSV preserves the same row order as the input file.
    """
    _require_artifacts()

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV files are accepted.")

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

    # Sort output by _orig_idx to restore input CSV row order
    df_proc = df_proc.sort_values("_orig_idx").reset_index(drop=True)

    out_df = df_proc[["date", "_currency_code_orig"]].rename(
        columns={"_currency_code_orig": "currency_code"}
    )
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
    """FIX Bug 5: Guard against both None predictor and unloaded artifacts."""
    if predictor is None or not predictor.artifacts_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run training pipeline first.",
        )
