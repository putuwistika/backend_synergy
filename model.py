# model.py
# Mongo + GridFS loader for SARIMAX Results, meta from train_df, and forecast wrapper.
from __future__ import annotations

import os
from io import BytesIO
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pymongo import MongoClient, DESCENDING
import gridfs

# ===================== ENV =====================
MONGODB_URI      = os.getenv("MONGODB_URI")  # REQUIRED (Railway Variables / local env)
DB_NAME          = os.getenv("DB_NAME", "forecasting_db")
GRIDFS_BUCKET    = os.getenv("GRIDFS_BUCKET", "models")
MODEL_FILENAME   = os.getenv("MODEL_FILENAME", "sarimax_model.pkl")
TRAIN_COLLECTION = os.getenv("TRAIN_COLLECTION", "train_df")
TEST_COLLECTION  = os.getenv("TEST_COLLECTION", "test_df")  # used by services.py
DEFAULT_FREQ     = os.getenv("DEFAULT_FREQ", "D").upper().strip()

# ===================== GLOBAL CACHES =====================
_mongo_client: Optional[MongoClient] = None
_db = None
_gfs: Optional[gridfs.GridFS] = None

_model_cache: Any = None
_model_file_id: Any = None  # GridFS _id for the cached model
_meta_cache: Optional[Dict[str, Any]] = None

# ===================== MONGO CONNECTION =====================
def _get_db():
    global _mongo_client, _db, _gfs
    if _db is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI is not set. Configure it in environment variables.")
        _mongo_client = MongoClient(MONGODB_URI)
        _db = _mongo_client[DB_NAME]
        _gfs = gridfs.GridFS(_db, collection=GRIDFS_BUCKET)
    return _db

def _get_gfs() -> gridfs.GridFS:
    if _gfs is None:
        _get_db()
    return _gfs  # type: ignore

# ===================== DATE UTILS =====================
def _parse_date_any(v) -> Optional[date]:
    """
    Accepts ISO string 'YYYY-MM-DD', Python datetime/date, or BSON date.
    """
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    s = str(v).strip()
    try:
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        return None

def _infer_freq_from_dates(dates: List[date]) -> str:
    """
    Very simple heuristic using median day-delta:
      ~1 -> 'D', ~7 -> 'W', >=25 -> 'M'
    """
    if len(dates) < 3:
        return DEFAULT_FREQ or "D"
    deltas = []
    for i in range(1, len(dates)):
        d = (dates[i] - dates[i - 1]).days
        if d > 0:
            deltas.append(d)
    if not deltas:
        return DEFAULT_FREQ or "D"
    med = int(np.median(deltas))
    if 6 <= med <= 8:
        return "W"
    if med >= 25:
        return "M"
    return "D"

def _days_in_month(y: int, m: int) -> int:
    if m in (1, 3, 5, 7, 8, 10, 12): return 31
    if m in (4, 6, 9, 11): return 30
    leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
    return 29 if leap else 28

def _add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, min(d.day, _days_in_month(y, m)))

# ===================== GRIDFS HELPERS =====================
def _latest_gridfs_file():
    """
    Fetch latest GridFS file document by filename.
    """
    fs = _get_gfs()
    cursor = fs.find({"filename": MODEL_FILENAME}).sort("uploadDate", DESCENDING).limit(1)
    for f in cursor:
        return f
    return None

def _load_results_from_bytes(blob: bytes):
    """
    Try official Statsmodels loaders from an in-memory buffer (supports .save() format),
    then fallback to raw pickle. Return object with .get_forecast() if possible.
    """
    # 1) SARIMAXResults.load from BytesIO
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults  # type: ignore
        buf = BytesIO(blob)
        obj = SARIMAXResults.load(buf)
        if hasattr(obj, "get_forecast"):
            return obj
    except Exception:
        pass

    # 2) ARIMAResults.load from BytesIO
    try:
        from statsmodels.tsa.arima.model import ARIMAResults  # type: ignore
        buf = BytesIO(blob)
        obj = ARIMAResults.load(buf)
        if hasattr(obj, "get_forecast"):
            return obj
    except Exception:
        pass

    # 3) Fallback: raw pickle (for pickle.dump of the results object)
    try:
        import pickle
        obj = pickle.loads(blob)
        if hasattr(obj, "get_forecast"):
            return obj
        if isinstance(obj, dict):
            for key in ("results", "res", "fitted", "fitted_results", "arima_results", "sarimax_results"):
                cand = obj.get(key)
                if hasattr(cand, "get_forecast"):
                    return cand
        return None
    except Exception:
        return None

def reload_model_from_gridfs() -> Dict[str, Any]:
    """
    Force reload model from GridFS into memory cache.
    Returns info about the loaded file.
    """
    global _model_cache, _model_file_id
    latest = _latest_gridfs_file()
    if latest is None:
        raise RuntimeError(f"gridfs_not_found: filename '{MODEL_FILENAME}' not found in bucket '{GRIDFS_BUCKET}.files'")

    blob = _get_gfs().get(latest._id).read()
    obj = _load_results_from_bytes(blob)
    if obj is None or not hasattr(obj, "get_forecast"):
        raise RuntimeError("unpickle_failed: Loaded object is not a statsmodels Results with get_forecast().")

    _model_cache = obj
    _model_file_id = latest._id
    return {
        "file_id": str(latest._id),
        "length": getattr(latest, "length", None),
        "uploadDate": getattr(latest, "uploadDate", None).isoformat() if getattr(latest, "uploadDate", None) else None,
        "filename": getattr(latest, "filename", None),
        "bucket": GRIDFS_BUCKET,
        "class_name": obj.__class__.__name__,
    }

def load_model() -> Dict[str, Any]:
    """
    Lazy-load model: if cache empty or GridFS file changed, reload.
    Returns info about the currently loaded file.
    """
    global _model_cache, _model_file_id
    latest = _latest_gridfs_file()
    if latest is None:
        raise RuntimeError(f"gridfs_not_found: filename '{MODEL_FILENAME}' not found in bucket '{GRIDFS_BUCKET}.files'")

    if _model_cache is None or str(_model_file_id) != str(latest._id):
        return reload_model_from_gridfs()
    # Already up to date
    return {
        "file_id": str(latest._id),
        "length": getattr(latest, "length", None),
        "uploadDate": getattr(latest, "uploadDate", None).isoformat() if getattr(latest, "uploadDate", None) else None,
        "filename": getattr(latest, "filename", None),
        "bucket": GRIDFS_BUCKET,
        "class_name": _model_cache.__class__.__name__ if _model_cache is not None else None,
    }

# ===================== META (from train_df in Mongo) =====================
def _detect_cols_from_sample(doc: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """
    Infer date & target columns from a sample document.
    """
    keys = [k for k in doc.keys() if k != "_id"]
    lower = {k.lower(): k for k in keys}

    date_col = lower.get("date") or lower.get("ds")
    if not date_col:
        for k in keys:
            if _parse_date_any(doc.get(k)) is not None:
                date_col = k
                break
    if not date_col:
        date_col = "Date"

    target_col = lower.get("revenue") or lower.get("y") or lower.get("value") or lower.get("target")
    if not target_col:
        for k in keys:
            if k == date_col:
                continue
            v = doc.get(k)
            try:
                float(str(v).replace(",", ""))
                target_col = k
                break
            except Exception:
                continue
    if not target_col:
        target_col = "Revenue"

    exog_cols = [k for k in keys if k not in {date_col, target_col}]
    return date_col, target_col, exog_cols

def _train_date_range(db, col_name: str, date_col: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Use aggregation to compute min/max on date_col (string ISO or BSON date).
    """
    col = db[col_name]
    pipeline = [
        {"$match": {date_col: {"$exists": True, "$ne": None}}},
        {"$group": {"_id": None, "min": {"$min": f"${date_col}"}, "max": {"$max": f"${date_col}"}}},
        {"$project": {"_id": 0, "min": 1, "max": 1}},
    ]
    res = list(col.aggregate(pipeline))
    if not res:
        return None, None
    dmin = _parse_date_any(res[0].get("min"))
    dmax = _parse_date_any(res[0].get("max"))
    return (dmin.isoformat() if dmin else None, dmax.isoformat() if dmax else None)

def _infer_freq_from_collection(db, col_name: str, date_col: str, sample_n: int = 200) -> str:
    """
    Take up to sample_n latest dates (ascending) and infer frequency.
    """
    col = db[col_name]
    cursor = col.find(
        {date_col: {"$exists": True, "$ne": None}},
        projection={date_col: 1, "_id": 0}
    ).sort(date_col, DESCENDING).limit(sample_n)
    dates_desc = [_parse_date_any(d[date_col]) for d in cursor]
    dates = [d for d in reversed(dates_desc) if d is not None]
    return _infer_freq_from_dates(dates) if dates else (DEFAULT_FREQ or "D")

def read_train_meta(force_reload: bool = False) -> Dict[str, Any]:
    """
    Build meta from train_df collection + model info:
      - date_col, target_col, exog_columns
      - train_range {start,end}
      - freq
      - model_order, model_seasonal_order
      - exog_names_from_model (if available)
    """
    global _meta_cache
    if _meta_cache is not None and not force_reload:
        return _meta_cache

    db = _get_db()
    col = db[TRAIN_COLLECTION]

    sample = col.find_one({}, projection={"_id": 0})
    if not sample:
        meta = {
            "date_col": "Date",
            "target_col": "Revenue",
            "exog_columns": [],
            "train_range": {"start": None, "end": None},
            "freq": DEFAULT_FREQ or "D",
            "model_order": None,
            "model_seasonal_order": None,
            "exog_names_from_model": None,
        }
        _meta_cache = meta
        return meta

    date_col, target_col, exog_cols = _detect_cols_from_sample(sample)
    dmin, dmax = _train_date_range(db, TRAIN_COLLECTION, date_col)
    freq = _infer_freq_from_collection(db, TRAIN_COLLECTION, date_col)

    # Ensure model is loaded (to read model attributes)
    try:
        load_model()
    except Exception:
        pass

    exog_names_from_model = None
    model_order = None
    model_seasonal_order = None
    if _model_cache is not None:
        try:
            model_order = tuple(getattr(_model_cache.model, "order", None)) if getattr(_model_cache.model, "order", None) else None
            model_seasonal_order = tuple(getattr(_model_cache.model, "seasonal_order", None)) if getattr(_model_cache.model, "seasonal_order", None) else None
        except Exception:
            pass
        try:
            exog_names = getattr(_model_cache.model, "exog_names", None)
            if exog_names and isinstance(exog_names, (list, tuple)):
                exog_names_from_model = list(exog_names)
                # Align exog_columns to training-time order
                exog_cols = list(exog_names_from_model)
        except Exception:
            pass

    meta = {
        "date_col": date_col,
        "target_col": target_col,
        "exog_columns": exog_cols,
        "train_range": {"start": dmin, "end": dmax},
        "freq": freq or (DEFAULT_FREQ or "D"),
        "model_order": model_order,
        "model_seasonal_order": model_seasonal_order,
        "exog_names_from_model": exog_names_from_model,
    }
    _meta_cache = meta
    return meta

# ===================== FUTURE DATES =====================
def next_dates_from_train(h: int, freq: str = "D") -> List[str]:
    """
    Build future dates after the last train date.
    """
    meta = read_train_meta()
    last = meta.get("train_range", {}).get("end")
    if last:
        start = datetime.fromisoformat(str(last)).date()
    else:
        start = datetime.utcnow().date()

    f = (freq or meta.get("freq") or "D").upper()
    if f == "W":
        return [(start + timedelta(days=7 * i)).isoformat() for i in range(1, h + 1)]
    if f == "M":
        return [_add_months(start, i).isoformat() for i in range(1, h + 1)]
    # default 'D'
    return [(start + timedelta(days=i)).isoformat() for i in range(1, h + 1)]

# ===================== FORECAST WRAPPER =====================
def _to_numpy_2d(ci_obj) -> np.ndarray:
    """
    Ensure conf_int result -> ndarray shape (h, 2)
    """
    if isinstance(ci_obj, np.ndarray):
        arr = ci_obj
    else:
        try:
            arr = ci_obj.to_numpy()
        except Exception:
            arr = np.asarray(ci_obj)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2]
    h = arr.shape[0] if arr.ndim >= 1 else 0
    return np.full((h, 2), np.nan, dtype=float)

def forecast(h: int, alpha: float = 0.05,
             exog_future: Optional[List[List[float]]] = None
             ) -> Tuple[List[float], List[float], List[float]]:
    """
    Run forecasting using the cached model from GridFS.
    Returns (mean, lower, upper) lists.
    """
    load_model()  # ensure cache up to date
    if _model_cache is None:
        raise RuntimeError("model_not_loaded: no model in cache")

    try:
        res = _model_cache.get_forecast(steps=h, exog=exog_future)
        mean = res.predicted_mean
        if hasattr(mean, "to_numpy"):
            mean = mean.to_numpy()
        else:
            mean = np.asarray(mean)
        ci = res.conf_int(alpha=alpha)
        ci_np = _to_numpy_2d(ci)
        lower = ci_np[:, 0] if ci_np.size else np.full(h, np.nan)
        upper = ci_np[:, 1] if ci_np.size else np.full(h, np.nan)
        return mean.astype(float).tolist(), lower.astype(float).tolist(), upper.astype(float).tolist()
    except Exception as e:
        import traceback; traceback.print_exc()
        raise RuntimeError(f"forecast_failed: {type(e).__name__}: {e}")
