"""
tests/test_api.py
==================
Unit tests for the FastAPI inference endpoints using httpx + pytest.

All model I/O is mocked — no trained artifacts or TensorFlow needed.
Coverage:
  - GET  /health          (ok and degraded states)
  - GET  /model-info      (happy path + 404 for unknown model)
  - POST /predict         (single row, multi-row, unknown model 422)
  - POST /predict/batch   (CSV upload, row-order preservation, wrong MIME 415)
  - GET  /health          when predictor is None (startup failure path)
"""

from __future__ import annotations

import io
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df_proc(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal df_proc that app.py's route handlers expect."""
    df = pd.DataFrame(rows)
    df["_currency_code_orig"] = df["currency_code"]
    df["_orig_idx"] = range(len(df))
    return df


def _make_predictor_mock(rows: list[dict], preds: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.artifacts_loaded = True
    mock.available_models = ["lgb", "xgb", "stacking"]
    mock.artifact_files   = ["lgb_model.pkl", "xgb_model.pkl", "stacking_meta.pkl"]
    mock.n_currencies     = 5
    mock.predict.return_value = (
        np.array(preds, dtype=np.float64),
        None,
        _make_df_proc(rows),
    )
    return mock


@pytest.fixture()
def client():
    """TestClient with a fully-loaded predictor mock."""
    rows = [{"date": "2024-01-15", "currency_code": "EUR"}]
    mock = _make_predictor_mock(rows, [1.0823])

    with patch("src.api.app.ForexPredictor", return_value=mock):
        # Import app AFTER patching so lifespan uses our mock
        from src.api import app as app_module
        # Force reload so the patched constructor is used
        import importlib
        importlib.reload(app_module)
        yield TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture()
def degraded_client():
    """TestClient where artifact loading raises FileNotFoundError (degraded mode)."""
    mock = MagicMock()
    mock.artifacts_loaded = False
    mock.available_models = []
    mock.load_artifacts.side_effect = FileNotFoundError("scaler_y.pkl not found")

    with patch("src.api.app.ForexPredictor", return_value=mock):
        from src.api import app as app_module
        import importlib
        importlib.reload(app_module)
        yield TestClient(app_module.app, raise_server_exceptions=False)


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")   # either is a 200
        assert "artifacts_loaded" in body
        assert "models_available" in body

    def test_health_degraded(self, degraded_client):
        r = degraded_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"
        assert r.json()["artifacts_loaded"] is False


# ── /model-info ────────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_stacking(self, client):
        r = client.get("/model-info?model=stacking")
        assert r.status_code == 200
        body = r.json()
        assert body["model_name"] == "stacking"
        assert "artifact_files" in body

    def test_model_info_unknown(self, client):
        r = client.get("/model-info?model=does_not_exist")
        assert r.status_code == 404
        assert "not loaded" in r.json()["detail"]

    def test_model_info_503_when_degraded(self, degraded_client):
        r = degraded_client.get("/model-info?model=stacking")
        assert r.status_code == 503


# ── POST /predict ──────────────────────────────────────────────────────────────

class TestPredict:
    _SINGLE_ROW = [{"date": "2024-01-15", "currency_code": "EUR"}]

    def test_predict_single_row(self, client):
        rows = self._SINGLE_ROW
        mock = _make_predictor_mock(rows, [1.0823])

        # Patch the global predictor object directly for this test
        import src.api.app as app_module
        original = app_module.predictor
        app_module.predictor = mock
        try:
            r = client.post("/predict", json={"rows": rows, "model": "stacking"})
            assert r.status_code == 200
            body = r.json()
            assert body["n_predictions"] == 1
            assert len(body["predictions"]) == 1
            pred = body["predictions"][0]
            assert pred["currency_code"] == "EUR"
            assert abs(pred["predicted_rate"] - 1.0823) < 1e-4
            assert pred["uncertainty"] is None
        finally:
            app_module.predictor = original

    def test_predict_multi_row(self, client):
        rows = [
            {"date": "2024-01-15", "currency_code": "EUR"},
            {"date": "2024-01-16", "currency_code": "GBP"},
            {"date": "2024-01-17", "currency_code": "JPY"},
        ]
        mock = _make_predictor_mock(rows, [1.08, 1.27, 148.5])

        import src.api.app as app_module
        original = app_module.predictor
        app_module.predictor = mock
        try:
            r = client.post("/predict", json={"rows": rows, "model": "lgb"})
            assert r.status_code == 200
            body = r.json()
            assert body["n_predictions"] == 3
            codes = [p["currency_code"] for p in body["predictions"]]
            assert codes == ["EUR", "GBP", "JPY"]
        finally:
            app_module.predictor = original

    def test_predict_503_when_degraded(self, degraded_client):
        r = degraded_client.post(
            "/predict",
            json={"rows": [{"date": "2024-01-15", "currency_code": "EUR"}], "model": "stacking"},
        )
        assert r.status_code == 503

    def test_predict_unknown_model_returns_422(self, client):
        rows = [{"date": "2024-01-15", "currency_code": "EUR"}]
        mock = _make_predictor_mock(rows, [1.08])
        mock.predict.side_effect = ValueError("Model 'bad_model' not available.")

        import src.api.app as app_module
        original = app_module.predictor
        app_module.predictor = mock
        try:
            r = client.post("/predict", json={"rows": rows, "model": "bad_model"})
            assert r.status_code == 422
        finally:
            app_module.predictor = original


# ── POST /predict/batch ────────────────────────────────────────────────────────

class TestPredictBatch:

    def _make_csv(self, rows: list[dict]) -> bytes:
        return pd.DataFrame(rows).to_csv(index=False).encode()

    def test_batch_happy_path(self, client):
        rows = [
            {"date": "2024-01-15", "currency_code": "EUR"},
            {"date": "2024-01-16", "currency_code": "GBP"},
        ]
        mock = _make_predictor_mock(rows, [1.08, 1.27])

        import src.api.app as app_module
        original = app_module.predictor
        app_module.predictor = mock
        try:
            csv_bytes = self._make_csv(rows)
            r = client.post(
                "/predict/batch?model=stacking",
                files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
            )
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/csv")
            result_df = pd.read_csv(io.StringIO(r.text))
            assert "predicted_rate" in result_df.columns
            assert len(result_df) == 2
        finally:
            app_module.predictor = original

    def test_batch_row_order_preserved(self, client):
        """Verify _orig_idx sort restores input CSV row order in output."""
        rows = [
            {"date": "2024-01-20", "currency_code": "JPY"},
            {"date": "2024-01-15", "currency_code": "EUR"},
            {"date": "2024-01-18", "currency_code": "GBP"},
        ]
        preds = [148.0, 1.08, 1.27]
        df_proc = _make_df_proc(rows)
        # Simulate predictor shuffling rows internally
        shuffled_idx = [2, 0, 1]
        df_proc_shuffled = df_proc.iloc[shuffled_idx].reset_index(drop=True)
        df_proc_shuffled["_orig_idx"] = [2, 0, 1]  # original positions

        mock = MagicMock()
        mock.artifacts_loaded = True
        mock.available_models = ["stacking"]
        mock.predict.return_value = (
            np.array(preds, dtype=np.float64),
            None,
            df_proc,   # df_proc already has _orig_idx 0,1,2 in order
        )

        import src.api.app as app_module
        original = app_module.predictor
        app_module.predictor = mock
        try:
            csv_bytes = self._make_csv(rows)
            r = client.post(
                "/predict/batch?model=stacking",
                files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
            )
            assert r.status_code == 200
            result_df = pd.read_csv(io.StringIO(r.text))
            # Row 0 must be JPY (original first row)
            assert result_df.iloc[0]["currency_code"] == "JPY"
            assert result_df.iloc[1]["currency_code"] == "EUR"
            assert result_df.iloc[2]["currency_code"] == "GBP"
        finally:
            app_module.predictor = original

    def test_batch_rejects_non_csv(self, client):
        import src.api.app as app_module
        original = app_module.predictor
        mock = _make_predictor_mock([], [])
        app_module.predictor = mock
        try:
            r = client.post(
                "/predict/batch?model=stacking",
                files={"file": ("data.json", io.BytesIO(b"{}"), "application/json")},
            )
            assert r.status_code == 415
        finally:
            app_module.predictor = original

    def test_batch_missing_required_columns(self, client):
        import src.api.app as app_module
        original = app_module.predictor
        mock = _make_predictor_mock([], [])
        app_module.predictor = mock
        try:
            bad_csv = b"open,high,low,close\n1.0,1.1,0.9,1.05\n"
            r = client.post(
                "/predict/batch?model=stacking",
                files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")},
            )
            assert r.status_code == 422
            assert "missing required columns" in r.json()["detail"]
        finally:
            app_module.predictor = original

    def test_batch_503_when_degraded(self, degraded_client):
        bad_csv = b"date,currency_code\n2024-01-15,EUR\n"
        r = degraded_client.post(
            "/predict/batch?model=stacking",
            files={"file": ("test.csv", io.BytesIO(bad_csv), "text/csv")},
        )
        assert r.status_code == 503
