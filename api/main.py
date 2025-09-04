# api/main.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from model import (
    read_train_meta,
    next_dates_from_train,
    forecast,
    load_model,                  # for /readyz
    reload_model_from_gridfs,    # for /admin/reload
)
from services import interpret_message, evaluate_on_test, build_smart_exog

# ================== ENV ==================
MONGODB_URI       = os.getenv("MONGODB_URI")
DB_NAME           = os.getenv("DB_NAME", "forecasting_db")
GRIDFS_BUCKET     = os.getenv("GRIDFS_BUCKET", "models")
MODEL_FILENAME    = os.getenv("MODEL_FILENAME", "sarimax_model.pkl")
TRAIN_COLLECTION  = os.getenv("TRAIN_COLLECTION", "train_df")
TEST_COLLECTION   = os.getenv("TEST_COLLECTION", "test_df")

DEFAULT_FREQ      = os.getenv("DEFAULT_FREQ", "D")
ALLOW_ORIGINS_ENV = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS     = [o.strip() for o in ALLOW_ORIGINS_ENV.split(",")] if ALLOW_ORIGINS_ENV else ["*"]

# ================== APP & CORS ==================
app = FastAPI(title="Forecast Backend (SARIMAX + Mongo GridFS)", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == ["*"] else ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Pydantic base ==================
class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

# ================== Schemas ==================
class Flags(BaseSchema):
    use_auto_exog: Optional[bool] = False
    exog_strategy: Optional[str] = "zeros"       # "zeros" | "smart"
    clip_non_negative: Optional[bool] = False    # clip output >= floor
    floor: Optional[float] = 0.0

class PredictRequest(BaseSchema):
    horizon: int = Field(..., ge=1, le=365)
    frequency: str = "D"
    alpha: float = 0.05
    # ✨ fleksibel: boleh map-style {col:[...]} atau structured {columns:[...], rows:[[...]]}
    exog: Optional[Any] = None
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
    mongo_info = _ping_mongo()
    model_info = {}
    try:
        model_info = load_model()  # lazy validate GridFS file
    except Exception as e:
        model_info = {"error": f"{type(e).__name__}: {e}"}

    meta_brief = {}
    try:
        m = read_train_meta(force_reload=True)
        meta_brief = {
            "date_col": m.get("date_col"),
            "target_col": m.get("target_col"),
            "exog_columns_len": len(m.get("exog_columns", []) or []),
            "exog_from_model_len": len(m.get("exog_names_from_model") or []),
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
    m = read_train_meta(force_reload=True)
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

# ================== Exog helpers ==================
def _expected_exog_cols(meta: dict) -> List[str]:
    """
    Ambil urutan exog dari model (training-time) kalau ada; LALU
    buang 'const'/'intercept' karena SARIMAX menambahkan konstanta internal.
    """
    names = meta.get("exog_names_from_model")
    if not names:
        names = meta.get("exog_columns", []) or []
    cleaned = [c for c in names if str(c).lower() not in ("const", "intercept")]
    return cleaned

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(",", ""))
        except Exception:
            return 0.0

def _flatten_series(v) -> List[float]:
    """
    Terima:
      - list 1D: [v1, v2, ...]
      - list 2D "baris": [[v1], [v2], ...] → flatten ambil elemen pertama
    """
    if not isinstance(v, list):
        return []
    if v and isinstance(v[0], list):
        return [_to_float(r[0]) if r else 0.0 for r in v]
    return [_to_float(z) for z in v]

def _align_exog_flexible(exog: Any, expected_cols: List[str], h: int):
    """
    Terima:
      - structured: {"columns":[...], "rows":[[...], ...]}
      - map-style:  { "<col>":[...], "<col2>":[...], ... }  (1D atau 2D di-flatten)
    Kembalikan: (rows_aligned, summary, warnings)
    """
    warnings: List[str] = []
    if exog is None:
        return None, {}, warnings

    # --- Structured {"columns","rows"} ---
    if isinstance(exog, dict) and "columns" in exog and "rows" in exog:
        cols_in = list(exog.get("columns") or [])
        rows_in = list(exog.get("rows") or [])
        if len(rows_in) != h:
            raise HTTPException(400, f"Exog rows must match horizon={h}")

        missing = [c for c in expected_cols if c not in cols_in]
        extra = [c for c in cols_in if c not in expected_cols]
        if missing:
            warnings.append(f"Exog columns missing and filled with 0: {missing}")
        if extra:
            warnings.append(f"Exog columns ignored (not used by model): {extra}")

        idx_map = {c: cols_in.index(c) for c in cols_in if c in expected_cols}
        aligned: List[List[float]] = []
        for r in rows_in:
            new_row = []
            for c in expected_cols:
                if c in idx_map:
                    j = idx_map[c]
                    try:
                        new_row.append(_to_float(r[j]))
                    except Exception:
                        new_row.append(0.0)
                else:
                    new_row.append(0.0)
            aligned.append(new_row)

        summary = {"columns": expected_cols, "source": "manual-structured", "missing_filled": missing, "extra_ignored": extra}
        return aligned, summary, warnings

    # --- Map-style {"col":[...], "col2":[...]} ---
    if isinstance(exog, dict):
        series_by_col: Dict[str, List[float]] = {}
        for k, v in exog.items():
            series_by_col[str(k)] = _flatten_series(v)

        missing = [c for c in expected_cols if c not in series_by_col]
        extra = [c for c in series_by_col.keys() if c not in expected_cols]
        if missing:
            warnings.append(f"Exog columns missing and filled with 0: {missing}")
        if extra:
            warnings.append(f"Exog columns ignored (not used by model): {extra}")

        aligned: List[List[float]] = []
        for i in range(h):
            row = []
            for c in expected_cols:
                seq = series_by_col.get(c)
                val = seq[i] if (seq is not None and i < len(seq)) else 0.0
                row.append(_to_float(val))
            aligned.append(row)

        summary = {"columns": expected_cols, "source": "manual-map", "missing_filled": missing, "extra_ignored": extra}
        return aligned, summary, warnings

    raise HTTPException(400, "Invalid exog format. Use {'columns','rows'} OR {'<col>':[...]} map.")

# ================== Routes (root & /api/* mirrors) ==================
@app.get("/")
def root():
    return {"ok": True, "service": "forecast-backend", "version": "0.7.0"}

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
    meta = read_train_meta(force_reload=True)

    # Kolom eksog yang dipakai model (drop const/intercept)
    expected_cols = _expected_exog_cols(meta)
    h = req.horizon
    freq = (req.frequency or meta.get("freq") or "D").upper()

    warnings: List[str] = []
    exog_summary: Dict[str, Any] = {}
    exog_future: Optional[List[List[float]]] = None
    mode = "none"

    # Precompute future dates (dipakai juga oleh smart-exog)
    dates = next_dates_from_train(h, freq=freq)

    if expected_cols:  # model butuh exog
        if req.flags and req.flags.use_auto_exog:
            strategy = (req.flags.exog_strategy or "zeros").lower()
            if strategy == "smart":
                exog_future, exog_summary, warns = build_smart_exog(h, expected_cols, dates)
                warnings.extend(warns)
            else:
                exog_future = [[0.0] * len(expected_cols) for _ in range(h)]
                exog_summary = {"mode": "zeros", "columns": expected_cols}
            mode = "auto"
        else:
            if not req.exog:
                raise HTTPException(
                    400,
                    "Model requires exogenous variables. Provide 'exog' or set flags.use_auto_exog=true."
                )
            exog_future, exog_summary, warns = _align_exog_flexible(req.exog, expected_cols, h)
            warnings.extend(warns)
            mode = "manual"
    else:
        mode = "none"
        if req.exog:
            warnings.append("Model does not expect exogenous variables; provided exog is ignored.")

    try:
        out = forecast(h=h, alpha=req.alpha, exog_future=exog_future)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if out is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or forecasting failed. Check Mongo GridFS artifacts & statsmodels compatibility."
        )

    mean, lower, upper = out

    # (Opsional) Clip output supaya tidak negatif (untuk tampilan)
    clip = bool(getattr(req.flags, "clip_non_negative", False))
    floor = float(getattr(req.flags, "floor", 0.0) or 0.0)
    if clip:
        mean = [max(floor, float(v)) for v in mean]
        if lower: lower = [max(floor, float(v)) for v in lower]
        if upper: upper = [max(floor, float(v)) for v in upper]
        warnings.append(f"Output clipped to >= {floor}. Prediction intervals no longer exact.")

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

# ================== Debug: expected exog & versions ==================
@app.get("/debug/exog")
@app.get("/api/debug/exog")
def debug_exog():
    m = read_train_meta(force_reload=True)
    raw_from_model = m.get("exog_names_from_model")
    cleaned = _expected_exog_cols(m)
    return {
        "expected_exog_from_model_raw": raw_from_model,   # bisa berisi 'const'
        "expected_exog_used_by_forecast": cleaned,        # tanpa 'const'/'intercept'
        "len_raw": (len(raw_from_model) if raw_from_model else 0),
        "len_used": len(cleaned),
        "freq": m.get("freq"),
        "train_range": m.get("train_range"),
    }

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
