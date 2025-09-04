# api/main.py
from __future__ import annotations

import os
import os.path as op
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from model import read_train_meta, next_dates_from_train, forecast
from services import build_auto_exog, interpret_message, evaluate_on_test

# ============== ENV ==============
DEFAULT_FREQ = os.getenv("DEFAULT_FREQ", "D")
MODEL_PATH   = os.getenv("MODEL_PATH", "artifacts/sarimax_model.pkl")
TRAIN_CSV    = os.getenv("TRAIN_CSV",  "artifacts/train_df.csv")
TEST_CSV     = os.getenv("TEST_CSV",   "artifacts/test_df.csv")

app = FastAPI(title="Forecast Backend (SARIMAX) — Vercel", version="0.3.0")


# --------- Schemas ----------
class Flags(BaseModel):
    use_auto_exog: Optional[bool] = False


class PredictRequest(BaseModel):
    horizon: int = Field(..., ge=1, le=365)
    frequency: str = "D"
    alpha: float = 0.05
    # exog manual (opsional): {"columns":[...], "rows":[[...], ...]}
    exog: Optional[Dict[str, List[List[float]]]] = None
    flags: Optional[Flags] = Flags()


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None


class PredictResponse(BaseModel):
    model_name: str
    generated_at: str
    horizon: int
    freq: str
    exog_mode: str  # "auto" | "manual" | "none"
    exog_summary: Dict[str, Any]
    forecasts: List[ForecastPoint]
    warnings: List[str]


class ChatReq(BaseModel):
    message: str
    alpha: float = 0.05


# --------- Internal helpers to register both "/" and "/api/*" ---------
def _healthz():
    return {"status": "ok"}


def _readyz():
    missing = [p for p in [MODEL_PATH, TRAIN_CSV, TEST_CSV] if not op.exists(p)]
    return {"ready": len(missing) == 0, "missing": missing}


def _meta():
    m = read_train_meta()
    exists = {
        "model": op.exists(MODEL_PATH),
        "train": op.exists(TRAIN_CSV),
        "test": op.exists(TEST_CSV),
    }
    return {
        "model_name": "sarimax",
        "default_freq": m.get("freq", DEFAULT_FREQ),
        "date_col": m.get("date_col"),
        "target_col": m.get("target_col"),
        "exog_columns": m.get("exog_columns", []),
        "train_range": m.get("train_range"),
        "model_order": m.get("model_order"),
        "model_seasonal_order": m.get("model_seasonal_order"),
        "artifacts_exist": exists,
    }


# --------- Routes (root & /api/*) ---------
@app.get("/")
def root():
    return {"ok": True, "service": "forecast-backend", "version": "0.3.0"}

@app.get("/healthz")
def healthz_root(): return _healthz()

@app.get("/api/healthz")
def healthz_api(): return _healthz()

@app.get("/readyz")
def readyz_root(): return _readyz()

@app.get("/api/readyz")
def readyz_api(): return _readyz()

@app.get("/meta")
def meta_root(): return _meta()

@app.get("/api/meta")
def meta_api(): return _meta()


# --------- Predict ---------
@app.post("/predict", response_model=PredictResponse)
@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    meta = read_train_meta()
    exog_cols = meta.get("exog_columns", [])
    h = req.horizon
    freq = (req.frequency or meta.get("freq") or "D").upper()

    warnings: List[str] = []
    exog_summary: Dict[str, Any] = {}
    exog_future: Optional[List[List[float]]] = None
    mode = "none"

    if exog_cols:
        mode = "manual"
        if req.flags and req.flags.use_auto_exog:
            exog_future, exog_summary, w = build_auto_exog(h, exog_cols)
            warnings.extend(w)
            mode = "auto"
        else:
            # Manual exog expected
            if not req.exog:
                raise HTTPException(
                    status_code=400,
                    detail="Model requires exogenous variables. Provide 'exog' or set flags.use_auto_exog=true."
                )
            cols = req.exog.get("columns", [])
            rows = req.exog.get("rows", [])
            if cols != exog_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Exog columns mismatch. Expected {exog_cols}, got {cols}"
                )
            if len(rows) != h:
                raise HTTPException(
                    status_code=400,
                    detail=f"Exog rows must match horizon={h}"
                )
            exog_future = rows
            exog_summary = {"columns": cols, "source": "manual"}

    out = forecast(h=h, alpha=req.alpha, exog_future=exog_future)
    if out is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or forecasting failed. Ensure artifacts/sarimax_model.pkl exists & is compatible with statsmodels."
        )
    mean, lower, upper = out
    dates = next_dates_from_train(h, freq=freq)

    points = [
        ForecastPoint(
            ds=dates[i],
            yhat=float(mean[i]),
            yhat_lower=float(lower[i]) if lower else None,
            yhat_upper=float(upper[i]) if upper else None,
        )
        for i in range(h)
    ]

    return PredictResponse(
        model_name="sarimax",
        generated_at=datetime.utcnow().isoformat() + "Z",
        horizon=h,
        freq=freq,
        exog_mode=mode,
        exog_summary=exog_summary,
        forecasts=points,
        warnings=warnings,
    )


# --------- Chat → Forecast ---------
@app.post("/chat/forecast")
@app.post("/api/chat/forecast")
def chat_forecast(body: ChatReq):
    parsed = interpret_message(body.message)
    req = PredictRequest(
        horizon=parsed["horizon"],
        frequency=parsed["frequency"],
        alpha=body.alpha,
        flags=parsed.get("flags")
    )
    return api_predict(req)  # delegate


# --------- Metrics (tanpa upload) ---------
@app.get("/metrics")
@app.get("/api/metrics")
def api_metrics(
    eval_start: str | None = Query(None, description="YYYY-MM-DD"),
    eval_end: str | None = Query(None, description="YYYY-MM-DD"),
    alpha: float = 0.05
):
    """
    Evaluasi performa pada TEST_CSV bawaan. Jika tidak ada/empty, akan return warnings.
    """
    return evaluate_on_test(alpha=alpha, eval_start=eval_start, eval_end=eval_end)
