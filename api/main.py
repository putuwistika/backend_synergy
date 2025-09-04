# api/main.py
from __future__ import annotations

import os
import os.path as op
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from model import (
    read_train_meta,
    next_dates_from_train,
    forecast,
    load_model,                  # untuk /readyz
    reload_model_from_gridfs,    # untuk /admin/reload
)
from services import build_auto_exog, interpret_message, evaluate_on_test

# ================== ENV ==================
MONGODB_URI      = os.getenv("MONGODB_URI")
DB_NAME          = os.getenv("DB_NAME", "forecasting_db")
GRIDFS_BUCKET    = os.getenv("GRIDFS_BUCKET", "models")
MODEL_FILENAME   = os.getenv("MODEL_FILENAME", "sarimax_model.pkl")
TRAIN_COLLECTION = os.getenv("TRAIN_COLLECTION", "train_df")
TEST_COLLECTION  = os.getenv("TEST_COLLECTION", "test_df")

DEFAULT_FREQ     = os.getenv("DEFAULT_FREQ", "D")
ALLOW_ORIGINS_ENV = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in ALLOW_ORIGINS_ENV.split(",")] if ALLOW_ORIGINS_ENV else ["*"]

# ================== APP & CORS ==================
app = FastAPI(title="Forecast Backend (SARIMAX + Mongo GridFS)", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == ["*"] else ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Pydantic Base (fix protected_namespaces) ==================
class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

# ================== Schemas ==================
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

# ================== Health helpers ==================
def _ping_mongo() -> Dict[str, Any]:
    """
    Cek koneksi Mongo + info ringan koleksi. Tidak melempar error;
    kembalikan status + error msg bila ada.
    """
    info: Dict[str, Any] = {"connected": False, "error": None, "collections": {}}
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000) if MONGODB_URI else None
        if client is None:
            info["error"] = "MONGODB_URI not set"
            return info
        client.admin.command("ping")
        db = client[DB_NAME]
        info["connected"] = True
        # gunakan estimated_document_count agar ringan
        try:
            info["collections"] = {
                "train_df_count": db[TRAIN_COLLECTION].estimated_document_count(),
                "test_df_count": db[TEST_COLLECTION].estimated_document_count(),
            }
        except Exception:
            pass
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info

def _healthz():
    return {"status": "ok"}

def _readyz():
    # 1) Mongo
    mongo_info = _ping_mongo()

    # 2) GridFS model (file id / length / uploadDate)
    model_info = {}
    try:
        model_info = load_model()  # lazy: muat / validasi file terbaru
    except Exception as e:
        model_info = {"error": f"{type(e).__name__}: {e}"}

    # 3) Meta ringkas
    meta_brief = {}
    try:
        m = read_train_meta()
        meta_brief = {
            "date_col": m.get("date_col"),
            "target_col": m.get("target_col"),
            "exog_columns_len": len(m.get("exog_columns", []) or []),
            "freq": m.get("freq"),
            "train_range": m.get("train_range"),
            "model_order": m.get("model_order"),
            "model_seasonal_order": m.get("model_seasonal_order"),
        }
    except Exception as e:
        meta_brief = {"error": f"{type(e).__name__}: {e}"}

    ready = (
        mongo_info.get("connected") is True
        and "error" not in model_info
        and "error" not in meta_brief
    )
    return {
        "ready": ready,
        "mongo": mongo_info,
        "gridfs_model": model_info,
        "meta": meta_brief,
        "config": {
            "db": DB_NAME,
            "bucket": GRIDFS_BUCKET,
            "model_filename": MODEL_FILENAME,
            "train_collection": TRAIN_COLLECTION,
            "test_collection": TEST_COLLECTION,
        },
    }

def _meta():
    m = read_train_meta()
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
    }

# ================== Routes (root & /api/* mirrors) ==================
@app.get("/")
def root():
    return {"ok": True, "service": "forecast-backend", "version": "0.4.0"}

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

# ================== Predict ==================
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

    try:
        out = forecast(h=h, alpha=req.alpha, exog_future=exog_future)
    except Exception as e:
        # bubble up detail (version mismatch, shape mismatch, dll)
        raise HTTPException(status_code=503, detail=str(e))

    if out is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or forecasting failed. Check Mongo GridFS artifacts & statsmodels compatibility."
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

# ================== Chat → Forecast ==================
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
    return api_predict(req)

# ================== Metrics ==================
@app.get("/metrics")
@app.get("/api/metrics")
def api_metrics(
    eval_start: str | None = Query(None, description="YYYY-MM-DD"),
    eval_end: str | None = Query(None, description="YYYY-MM-DD"),
    alpha: float = 0.05
):
    return evaluate_on_test(alpha=alpha, eval_start=eval_start, eval_end=eval_end)

# ================== Admin (force reload model dari GridFS) ==================
@app.post("/admin/reload")
@app.post("/api/admin/reload")
def admin_reload_model():
    try:
        info = reload_model_from_gridfs()
        return {"reloaded": True, "gridfs_model": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================== Debug Versions ==================
from importlib.metadata import version as _pkgver, PackageNotFoundError
import sys
def _v(pkg: str):
    try: return _pkgver(pkg)
    except PackageNotFoundError: return None

@app.get("/debug/versions")
@app.get("/api/debug/versions")
def debug_versions():
    return {
        "python": sys.version,
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "pandas": _v("pandas"),
        "statsmodels": _v("statsmodels"),
        "pymongo": _v("pymongo"),
        "fastapi": _v("fastapi"),
        "pydantic": _v("pydantic"),
    }
