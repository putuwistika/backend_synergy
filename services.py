# services.py
# Mongo-ready services: auto-exog, chat interpretation, and test-set metrics
from __future__ import annotations

import os
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from pymongo import MongoClient, ASCENDING

from model import read_train_meta, forecast  # pakai meta & forecast dari model.py

# ===================== ENV & Mongo =====================
MONGODB_URI      = os.getenv("MONGODB_URI")
DB_NAME          = os.getenv("DB_NAME", "forecasting_db")
TEST_COLLECTION  = os.getenv("TEST_COLLECTION", "test_df")

_client: Optional[MongoClient] = None
_db = None

def _get_db():
    global _client, _db
    if _db is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI is not set. Please configure it in your environment.")
        _client = MongoClient(MONGODB_URI)
        _db = _client[DB_NAME]
    return _db

# ===================== Helpers =====================
_EPS = 1e-12

def _parse_date_any(v) -> Optional[date]:
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

def _to_float_or_zero(v) -> float:
    try:
        # izinkan string bernilai numerik
        return float(v)
    except Exception:
        try:
            return float(str(v).replace(",", ""))
        except Exception:
            return 0.0

# ===================== AUTO-EXOG =====================
def build_auto_exog(h: int, exog_cols: List[str]) -> Tuple[List[List[float]], Dict[str, Any], List[str]]:
    """
    Baseline ringan: isi 0.0 untuk semua kolom exog (aman untuk 'tanpa exog').
    """
    warnings: List[str] = []
    if exog_cols:
        warnings.append("Exogenous variables auto-filled with zeros (baseline).")
    X = [[0.0 for _ in exog_cols] for _ in range(h)]
    summary = {"mode": "zeros", "columns": exog_cols}
    return X, summary, warnings

# ===================== CHAT INTERPRETER =====================
def interpret_message(msg: str) -> Dict[str, Any]:
    """
    Contoh yang dikenali:
      - "forecast 15 hari", "forecast 6 minggu", "forecast 3 bulan"
      - "tanpa exog" -> flags.use_auto_exog = True
    Default: 14 hari (D)
    """
    txt = (msg or "").strip().lower()
    import re

    horizon, freq = 14, "D"
    m = re.search(r"forecast\s+(\d+)\s*(hari|minggu|bulan|day|week|month)?", txt)
    if m:
        n = int(m.group(1))
        unit = (m.group(2) or "").strip()
        if unit in ("", "hari", "day"):
            horizon, freq = n, "D"
        elif unit in ("minggu", "week"):
            horizon, freq = n, "W"
        elif unit in ("bulan", "month"):
            horizon, freq = n, "M"
        else:
            horizon, freq = n, "D"

    flags = {}
    if "tanpa exog" in txt or "no exog" in txt or "auto exog" in txt:
        flags = {"use_auto_exog": True}

    return {
        "intent": "forecast",
        "horizon": max(1, min(365, horizon)),
        "frequency": freq,
        "flags": flags
    }

# ===================== METRICS (Mongo test_df) =====================
def _read_test_from_mongo(meta: Dict[str, Any],
                          eval_start: Optional[str],
                          eval_end: Optional[str]) -> Tuple[List[str], List[float], Dict[str, List[float]]]:
    """
    Tarik window test_df (urut ascending by date) dari Mongo dengan proyeksi kolom minimum.
    Return:
        dates: List[str iso]
        y:     List[float]
        exog_by_col: Dict[col, List[float]]
    """
    db = _get_db()
    date_col = meta.get("date_col", "Date")
    target_col = meta.get("target_col", "Revenue")
    exog_cols: List[str] = meta.get("exog_columns", [])

    # Build query
    q: Dict[str, Any] = {date_col: {"$exists": True, "$ne": None}}
    # Range filter jika diberikan
    range_q: Dict[str, Any] = {}
    if eval_start:
        range_q["$gte"] = eval_start
    if eval_end:
        range_q["$lte"] = eval_end
    if range_q:
        q[date_col] = {**q[date_col], **range_q}

    # Projection
    proj = {date_col: 1, target_col: 1, "_id": 0}
    for c in exog_cols:
        proj[c] = 1

    cursor = db[TEST_COLLECTION].find(q, projection=proj).sort(date_col, ASCENDING)

    dates: List[str] = []
    y_vals: List[float] = []
    exog_by_col: Dict[str, List[float]] = {c: [] for c in exog_cols}

    for doc in cursor:
        d = _parse_date_any(doc.get(date_col))
        if not d:
            continue
        dates.append(d.isoformat())

        y_raw = doc.get(target_col)
        y_vals.append(_to_float_or_zero(y_raw))

        for c in exog_cols:
            ex = _to_float_or_zero(doc.get(c))
            exog_by_col[c].append(ex)

    return dates, y_vals, exog_by_col

def _slice_by_date_range(dates: List[str], start: Optional[str], end: Optional[str]) -> Tuple[int, int]:
    """
    dates di sini sudah ascending. Ambil indeks [i0, i1) untuk rentang inklusif start..end.
    """
    if not dates:
        return 0, 0
    i0, i1 = 0, len(dates)
    if start:
        for i, ds in enumerate(dates):
            if ds >= start:
                i0 = i
                break
    if end:
        j = i1
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= end:
                j = i + 1
                break
        i1 = max(i0, j)
    return i0, i1

def _metrics_basic(y_true: np.ndarray, y_hat: np.ndarray,
                   lower: Optional[np.ndarray], upper: Optional[np.ndarray]) -> Dict[str, Any]:
    err = y_hat - y_true
    mae = float(np.nanmean(np.abs(err)))
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    mape = float(np.nanmean(np.abs(err) / np.maximum(np.abs(y_true), _EPS)))
    smape = float(np.nanmean(2.0 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_hat), _EPS)))
    bias = float(np.nanmean(err))
    coverage = None
    if lower is not None and upper is not None:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = float(np.nanmean(inside.astype(np.float64)))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "bias_me": bias,
        "coverage_95": coverage
    }

def evaluate_on_test(alpha: float = 0.05,
                     eval_start: Optional[str] = None,
                     eval_end: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluasi performa pada koleksi test_df di Mongo.
    - Susun exog sesuai urutan meta['exog_columns'] (yang diselaraskan dengan model.exog_names kalau ada).
    - Panggil forecast untuk n langkah dan hitung metrik.
    """
    meta = read_train_meta()
    exog_cols: List[str] = meta.get("exog_columns", [])

    dates_all, y_all, exog_by_col = _read_test_from_mongo(meta, eval_start, eval_end)
    if not dates_all:
        return {
            "eval_window": {"start": eval_start, "end": eval_end, "n": 0},
            "metrics": None,
            "by_period": [],
            "warnings": ["No test documents found in MongoDB."]
        }

    # window indeks (aman jika pengguna mengirim start/end yang tidak pas)
    i0, i1 = _slice_by_date_range(dates_all, eval_start, eval_end)
    dates = dates_all[i0:i1]
    n = len(dates)
    if n == 0:
        return {
            "eval_window": {"start": eval_start, "end": eval_end, "n": 0},
            "metrics": None,
            "by_period": [],
            "warnings": ["Selected evaluation window has no rows."]
        }

    y_true = np.array(y_all[i0:i1], dtype=float)

    # Rakit exog_future dengan urutan tepat
    exog_future: Optional[List[List[float]]] = None
    if exog_cols:
        X: List[List[float]] = []
        for t in range(i0, i1):
            row: List[float] = []
            for c in exog_cols:
                vals = exog_by_col.get(c, [])
                val = vals[t] if t < len(vals) else 0.0
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                row.append(float(val))
            X.append(row)
        exog_future = X

    # Forecast n langkah
    out = forecast(h=n, alpha=alpha, exog_future=exog_future)
    if out is None:
        return {
            "eval_window": {"start": dates[0], "end": dates[-1], "n": n},
            "metrics": None,
            "by_period": [],
            "warnings": ["Model not loaded or forecasting failed."]
        }

    mean, lower, upper = out
    y_hat = np.array(mean, dtype=float)
    lo = np.array(lower, dtype=float) if lower else None
    hi = np.array(upper, dtype=float) if upper else None

    # Metrik agregat
    m = _metrics_basic(y_true, y_hat, lo, hi)

    # Detail per tanggal (untuk tabel FE)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rows.append({
            "ds": dates[i],
            "y": float(y_true[i]) if not np.isnan(y_true[i]) else None,
            "yhat": float(y_hat[i]) if not np.isnan(y_hat[i]) else None,
            "lower": (float(lo[i]) if lo is not None and not np.isnan(lo[i]) else None),
            "upper": (float(hi[i]) if hi is not None and not np.isnan(hi[i]) else None),
            "abs_err": (abs(float(y_hat[i] - y_true[i])) if not (np.isnan(y_hat[i]) or np.isnan(y_true[i])) else None)
        })

    return {
        "eval_window": {"start": dates[0], "end": dates[-1], "n": n},
        "metrics": m,
        "by_period": rows,
        "warnings": []
    }
