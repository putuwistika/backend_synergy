# services.py
# Utilities for auto-exog, chat interpretation, and metrics evaluation
from __future__ import annotations

import os
import csv
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

from model import read_train_meta, forecast

# ===== ENV =====
TRAIN_CSV = os.getenv("TRAIN_CSV", "artifacts/train_df.csv")
TEST_CSV  = os.getenv("TEST_CSV",  "artifacts/test_df.csv")

_EPS = 1e-12


# ===================== AUTO-EXOG =====================
def build_auto_exog(h: int, exog_cols: List[str]) -> Tuple[List[List[float]], Dict[str, Any], List[str]]:
    """
    Baseline super-ringan: isi 0 untuk semua kolom exogenous.
    Aman untuk one-hot/flag/numeric ketika user minta 'tanpa exog'.
    """
    warnings: List[str] = []
    summary: Dict[str, Any] = {"mode": "zeros", "columns": exog_cols, "note": "auto-filled"}
    if exog_cols:
        warnings.append("Exogenous variables auto-filled with zeros (baseline).")
    X = [[0.0 for _ in exog_cols] for _ in range(h)]
    return X, summary, warnings


# ===================== CHAT INTERPRETER =====================
def interpret_message(msg: str) -> Dict[str, Any]:
    """
    Parser sederhana untuk kalimat natural Indo/EN:
    - "forecast 15 hari", "forecast 6 minggu", "forecast 3 bulan"
    - "tanpa exog" → flags.use_auto_exog = True
    Default: horizon=14, freq="D"
    """
    txt = msg.strip().lower()
    import re

    horizon = 14
    freq = "D"

    # angka + satuan
    m = re.search(r"forecast\s+(\d+)\s*(hari|minggu|bulan|day|week|month)?", txt)
    if m:
        n = int(m.group(1))
        unit = (m.group(2) or "").strip()
        if unit in ("hari", "day", ""):
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


# ===================== CSV HELPERS =====================
def _parse_iso_date(s: str) -> Optional[date]:
    try:
        return datetime.fromisoformat(s.strip()[:10]).date()
    except Exception:
        return None


def _load_csv_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            header = []
    return header


def _read_test_rows(meta: Dict[str, Any]) -> Tuple[List[str], List[float], Dict[str, List[float]]]:
    """
    Baca TEST_CSV -> (dates[], target[], exog_by_col{col: list[...]})
    *Tanpa pandas*, menggunakan csv module.
    """
    if not os.path.exists(TEST_CSV):
        return [], [], {c: [] for c in meta.get("exog_columns", [])}

    date_col = meta.get("date_col", "Date")
    target_col = meta.get("target_col", "Revenue")
    exog_cols: List[str] = meta.get("exog_columns", [])

    with open(TEST_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        dates: List[str] = []
        ys: List[float] = []
        exog_by_col: Dict[str, List[float]] = {c: [] for c in exog_cols}

        for row in r:
            # Tanggal
            d = _parse_iso_date(row.get(date_col, "") or row.get("ds", "") or "")
            if not d:
                continue
            dates.append(d.isoformat())

            # Target
            y_str = (row.get(target_col) or row.get("y") or "").strip()
            try:
                ys.append(float(y_str))
            except Exception:
                ys.append(np.nan)

            # Exog sesuai urutan meta
            for c in exog_cols:
                val_str = (row.get(c) or "").strip()
                try:
                    exog_by_col[c].append(float(val_str))
                except Exception:
                    exog_by_col[c].append(0.0)  # fallback aman

    return dates, ys, exog_by_col


def _slice_by_date_range(dates: List[str], start: Optional[str], end: Optional[str]) -> Tuple[int, int]:
    """
    Ambil indeks [i_start, i_end) untuk rentang tanggal inklusif start..end.
    Jika start atau end None -> otomatis ke ujung.
    """
    if not dates:
        return 0, 0
    # asumsi dates ascending (kebanyakan CSV test memang ascending)
    i0, i1 = 0, len(dates)

    if start:
        # first index >= start
        for i, ds in enumerate(dates):
            if ds >= start:
                i0 = i
                break
    if end:
        # last index <= end  -> exclusive upper bound
        j = i1
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= end:
                j = i + 1
                break
        i1 = max(i0, j)
    return i0, i1


# ===================== METRICS =====================
def _metrics_basic(y_true: np.ndarray, y_hat: np.ndarray, lower: Optional[np.ndarray], upper: Optional[np.ndarray]) -> Dict[str, Any]:
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


def evaluate_on_test(alpha: float = 0.05, eval_start: Optional[str] = None, eval_end: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluasi performa pada TEST_CSV bawaan (tanpa upload).
    Asumsi umum: periode test menerus dari akhir train; kita panggil forecast(h=len(window), exog=exog_window).
    """
    meta = read_train_meta()
    exog_cols: List[str] = meta.get("exog_columns", [])

    # Load test rows
    dates_all, y_all, exog_by_col = _read_test_rows(meta)
    if not dates_all:
        return {
            "eval_window": {"start": None, "end": None, "n": 0},
            "metrics": None,
            "by_period": [],
            "warnings": ["No TEST_CSV found or empty."]
        }

    # Slice by date range if provided
    i0, i1 = _slice_by_date_range(dates_all, eval_start, eval_end)
    dates = dates_all[i0:i1]
    y_true = np.array(y_all[i0:i1], dtype=float)
    n = len(dates)
    if n == 0:
        return {
            "eval_window": {"start": eval_start, "end": eval_end, "n": 0},
            "metrics": None,
            "by_period": [],
            "warnings": ["Selected evaluation window has no rows."]
        }

    # Prepare exog_future (n x k) if necessary
    exog_future: Optional[List[List[float]]] = None
    if exog_cols:
        X = []
        for t in range(i0, i1):
            row_vals = []
            for c in exog_cols:
                vals = exog_by_col.get(c, [])
                val = vals[t] if t < len(vals) else 0.0
                # guard NaN
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                row_vals.append(float(val))
            X.append(row_vals)
        exog_future = X

    # Forecast for n steps
    out = forecast(h=n, alpha=alpha, exog_future=exog_future)
    if out is None:
        return {
            "eval_window": {"start": dates[0], "end": dates[-1], "n": n},
            "metrics": None,
            "by_period": [],
            "warnings": ["Model not loaded or forecasting failed (check model artifact & exog columns)."]
        }

    mean, lower, upper = out
    y_hat = np.array(mean, dtype=float)
    lo = np.array(lower, dtype=float) if lower else None
    hi = np.array(upper, dtype=float) if upper else None

    # Metrics
    m = _metrics_basic(y_true, y_hat, lo, hi)

    # by_period
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
