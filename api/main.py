# api/main.py
from __future__ import annotations

import os
import os.path as op
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from model import read_train_meta, next_dates_from_train, forecast
from services import build_auto_exog, interpret_message, evaluate_on_test

# ================= ENV =================
DEFAULT_FREQ = os.getenv("DEFAULT_FREQ", "D")
MODEL_PATH   = os.getenv("MODEL_PATH", "artifacts/sarimax_model.pkl")
TRAIN_CSV    = os.getenv("TRAIN_CSV",  "artifacts/train_df.csv")
TEST_CSV     = os.getenv("TEST_CSV",   "artifacts/test_df.csv")
ALLOW_ORIGINS_ENV = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in ALLOW_ORIGINS_ENV.split(",")] if ALLOW_ORIGINS_ENV else ["*"]

app = FastAPI(title="Forecast Backend (SARIMAX) — Railway", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == ["*"] else ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Pydantic Base (fix warning protected_namespaces) =================
class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow fields like model_name


# ================= Schemas =================
class Flags(BaseSchema):
    use_auto_exog: Optional[bool] = False

class PredictRequest(BaseSchema):
    horizon: int = Field(..., ge=1, le=365)
    frequency: str = "D"
    alpha: float = 0.05
    # exog manual (opsional): {"columns":[...], "rows":[[...], ...]}
    exog: Optional[Dict[str, List[List[float]]]] = None
    flags: Optional[Flags] = Flags()

class ForecastPoint(BaseSchema):
    ds: str
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None

class PredictResponse(BaseSchema):
    model_name: str
    generated_at: str
    horizon: int
    freq: str
    exog_mode: str  # "auto" | "manual" | "none"
    exog_summary: Dict[str, Any]
    forecasts: List[ForecastPoint]
    warnings: List[str]

class ChatReq(BaseSchema):
    message: str
    alpha: float = 0.05


# ================= Internal helpers (shared handlers) =================
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
        "exog_columns_from_model": m.get("exog_names_from_model"),
        "train_range": m.get("train_range"),
        "model_order": m.get("model_order"),
        "model_seasonal_order": m.get("model_seasonal_order"),
        "artifacts_exist": exists,
    }


# ================= Routes (root & /api/* mirrors) =================
@app.get("/")
def root():
    return {"ok": True, "service": "forecast-backend", "version": "0.3.1"}

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


# ================= Predict =================
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
            # Manual exog expected: {"columns":[...], "rows":[[...], ...]}
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
            detail="Model not loaded or forecasting failed. Check artifacts & statsmodels compatibility."
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


# ================= Chat → Forecast (Flow C) =================
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
    return api_predict(req)  # delegate ke handler di atas


# ================= Metrics (tanpa upload) =================
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
