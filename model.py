# model.py
from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pymongo import MongoClient, ASCENDING, DESCENDING
import gridfs

# ===================== ENV =====================
MONGODB_URI      = os.getenv("MONGODB_URI")  # wajib di Railway
DB_NAME          = os.getenv("DB_NAME", "forecasting_db")
GRIDFS_BUCKET    = os.getenv("GRIDFS_BUCKET", "models")
MODEL_FILENAME   = os.getenv("MODEL_FILENAME", "sarimax_model.pkl")
TRAIN_COLLECTION = os.getenv("TRAIN_COLLECTION", "train_df")
TEST_COLLECTION  = os.getenv("TEST_COLLECTION", "test_df")  # dipakai di services.py
DEFAULT_FREQ     = os.getenv("DEFAULT_FREQ", "D").upper().strip()

# ===================== GLOBAL CACHES =====================
_mongo_client: Optional[MongoClient] = None
_db = None
_gfs: Optional[gridfs.GridFS] = None

_model_cache: Any = None
_model_file_id: Any = None  # GridFS file _id of cached model
_meta_cache: Optional[Dict[str, Any]] = None


# ===================== UTIL: MONGO CONN =====================
def _get_db():
    global _mongo_client, _db, _gfs
    if _db is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI is not set. Please set it in your environment.")
        _mongo_client = MongoClient(MONGODB_URI)
        _db = _mongo_client[DB_NAME]
        _gfs = gridfs.GridFS(_db, collection=GRIDFS_BUCKET)
    return _db


def _get_gfs() -> gridfs.GridFS:
    if _gfs is None:
        _get_db()
    return _gfs  # type: ignore


# ===================== UTIL: DATES =====================
def _parse_date_any(v) -> Optional[date]:
    """
    Terima string ISO 'YYYY-MM-DD' atau datetime (BSON date dari Mongo).
    """
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    # string
    s = str(v).strip()
    # ambil 10 char pertama (YYYY-MM-DD) untuk aman
    try:
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        return None


def _infer_freq_from_dates(dates: List[date]) -> str:
    """
    Heuristik sederhana dari median delta-hari:
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
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m in (4, 6, 9, 11):
        return 30
    # February
    leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
    return 29 if leap else 28


def _add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, min(d.day, _days_in_month(y, m)))


# ===================== LOAD MODEL (GridFS) =====================
def _latest_gridfs_file():
    """
    Ambil file GridFS terbaru berdasarkan filename.
    """
    fs = _get_gfs()
    cursor = fs.find({"filename": MODEL_FILENAME}).sort("uploadDate", DESCENDING).limit(1)
    for f in cursor:
        return f
    return None


def _load_results_from_bytes(blob: bytes):
    """
    Unpickle menjadi SARIMAXResults/ARIMAResults.
    """
    # 1) SARIMAXResults
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults  # type: ignore
        import pickle
        obj = pickle.loads(blob)
        # obj bisa langsung Results; kalau bukan, biarkan exception di get_forecast nanti
        if hasattr(obj, "get_forecast"):
            return obj
        # beberapa format pickle adalah dict { 'results': Results }
        if isinstance(obj, dict) and "results" in obj and hasattr(obj["results"], "get_forecast"):
            return obj["results"]
        return obj
    except Exception:
        pass

    # 2) ARIMAResults
    try:
        from statsmodels.tsa.arima.model import ARIMAResults  # type: ignore
        import pickle
        obj = pickle.loads(blob)
        if hasattr(obj, "get_forecast"):
            return obj
        return obj
    except Exception:
        import pickle
        try:
            return pickle.loads(blob)  # last resort
        except Exception:
            return None


def reload_model_from_gridfs() -> Dict[str, Any]:
    """
    Force reload model dari GridFS, isi cache.
    Return informasi file (id, length, uploadDate).
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
        "length": latest.length,
        "uploadDate": latest.uploadDate.isoformat() if hasattr(latest, "uploadDate") else None,
        "filename": latest.filename,
        "bucket": GRIDFS_BUCKET,
    }


def load_model():
    """
    Lazy-load model: jika cache kosong atau file GridFS terbaru beda, reload.
    """
    global _model_cache, _model_file_id
    latest = _latest_gridfs_file()
    if latest is None:
        raise RuntimeError(f"gridfs_not_found: filename '{MODEL_FILENAME}' not found in bucket '{GRIDFS_BUCKET}.files'")

    if _model_cache is None or str(_model_file_id) != str(latest._id):
        # reload
        return reload_model_from_gridfs()
    # sudah up-to-date
    return {
        "file_id": str(latest._id),
        "length": latest.length,
        "uploadDate": latest.uploadDate.isoformat() if hasattr(latest, "uploadDate") else None,
        "filename": latest.filename,
        "bucket": GRIDFS_BUCKET,
    }


# ===================== META (dari train_df di Mongo) =====================
def _detect_cols_from_sample(doc: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """
    Deteksi date & target col dari dokumen contoh.
    """
    keys = [k for k in doc.keys() if k != "_id"]
    lower = {k.lower(): k for k in keys}

    date_col = lower.get("date") or lower.get("ds")
    if not date_col:
        # cari field pertama yang bisa diparse ke date
        for k in keys:
            if _parse_date_any(doc.get(k)) is not None:
                date_col = k
                break
    if not date_col:
        date_col = "Date"

    target_col = lower.get("revenue") or lower.get("y") or lower.get("value") or lower.get("target")
    if not target_col:
        # ambil numeric non-date pertama
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
    Gunakan aggregation untuk min/max tanggal (string ISO atau BSON date).
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
    Ambil hingga sample_n tanggal terakhir (ascending), infer freq.
    """
    col = db[col_name]
    # ambil ascending dari tail: sort desc limit sample_n, lalu balik
    cursor = col.find(
        {date_col: {"$exists": True, "$ne": None}},
        projection={date_col: 1, "_id": 0}
    ).sort(date_col, DESCENDING).limit(sample_n)
    dates_desc = [_parse_date_any(d[date_col]) for d in cursor]
    dates = [d for d in reversed(dates_desc) if d is not None]
    return _infer_freq_from_dates(dates) if dates else (DEFAULT_FREQ or "D")


def read_train_meta(force_reload: bool = False) -> Dict[str, Any]:
    """
    Baca meta dari koleksi train_df + info model.
    - date_col, target_col, exog_columns
    - train_range {start,end}
    - freq
    - model_order, model_seasonal_order
    - exog_names_from_model (jika ada)
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

    # range tanggal & freq
    dmin, dmax = _train_date_range(db, TRAIN_COLLECTION, date_col)
    freq = _infer_freq_from_collection(db, TRAIN_COLLECTION, date_col)

    # info dari model
    # pastikan model sudah termuat (atau coba load sekali)
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
                # sinkronkan urutan exog ke urutan saat training
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
    Bangun tanggal masa depan sepanjang h berdasarkan last date di train_df.
    """
    meta = read_train_meta()
    last = meta.get("train_range", {}).get("end")
    if last:
        start = datetime.fromisoformat(str(last)).date()
    else:
        start = datetime.utcnow().date()

    f = (freq or meta.get("freq") or "D").upper()
    out: List[str] = []
    if f == "W":
        out = [(start + timedelta(days=7 * i)).isoformat() for i in range(1, h + 1)]
    elif f == "M":
        out = [_add_months(start, i).isoformat() for i in range(1, h + 1)]
    else:
        out = [(start + timedelta(days=i)).isoformat() for i in range(1, h + 1)]
    return out


# ===================== FORECAST WRAPPER =====================
def _to_numpy_2d(ci_obj) -> np.ndarray:
    """
    Pastikan conf_int -> ndarray shape (h,2)
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
    Jalankan forecasting dengan model dari GridFS.
    Return (mean, lower, upper) sebagai list float.
    """
    # pastikan model cache up-to-date
    load_model()
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
