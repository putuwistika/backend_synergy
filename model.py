"""
model.py
Utility untuk:
- Load artefak SARIMAX/ARIMA Results (pickle)
- Baca meta dari train_df.csv (date_col, target_col, exog_columns, train_range)
- Infer frekuensi sederhana (D/W/M) bila perlu
- Generate tanggal future dan jalankan forecast + confidence interval

Catatan:
- Tidak menggunakan pandas agar bundle kecil di Vercel.
- Kompatibel dengan statsmodels 0.14.x; conf_int bisa berupa np.ndarray atau
  DataFrame-like (punya .to_numpy()) — keduanya ditangani.
"""

from __future__ import annotations

import os
import csv
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ============== ENV ==============
MODEL_PATH   = os.getenv("MODEL_PATH", "artifacts/sarimax_model.pkl")
TRAIN_CSV    = os.getenv("TRAIN_CSV",  "artifacts/train_df.csv")
DEFAULT_FREQ = os.getenv("DEFAULT_FREQ", "D").upper().strip()  # "D"|"W"|"M"


# ============== INTERNAL CACHE ==============
_model_cache: Any = None
_meta_cache: Optional[Dict[str, Any]] = None


# ============== LOADER ==============
def _load_results(path: str):
    """
    Coba load artefak sebagai SARIMAXResults, fallback ARIMAResults.
    """
    # 1) SARIMAXResults
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults  # type: ignore
        return SARIMAXResults.load(path)
    except Exception:
        pass

    # 2) ARIMAResults (statsmodels ARIMA modern)
    try:
        from statsmodels.tsa.arima.model import ARIMAResults  # type: ignore
        return ARIMAResults.load(path)
    except Exception:
        return None


def load_model():
    """Lazy-load model results sekali, cache di memori Lambda."""
    global _model_cache
    if _model_cache is None and os.path.exists(MODEL_PATH):
        _model_cache = _load_results(MODEL_PATH)
    return _model_cache


# ============== CSV META PARSER ==============
def _ci_find(header: List[str], candidates: List[str]) -> Optional[str]:
    """
    Case-insensitive find kolom pertama yang cocok di header.
    """
    h_lower = [h.lower() for h in header]
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in h_lower:
            return header[h_lower.index(cand_l)]
    return None


def _detect_cols(header: List[str]) -> Tuple[str, str, List[str]]:
    """
    Deteksi kolom tanggal & target; sisanya dianggap exog.
    Preferensi nama umum: Date/ds, Revenue/y.
    """
    if not header:
        # fallback aman
        return "Date", "Revenue", []

    date_col = _ci_find(header, ["Date", "ds"]) or header[0]
    tgt_col  = _ci_find(header, ["Revenue", "y", "value", "target"]) or (
        header[1] if len(header) > 1 else header[0]
    )
    exog_cols = [c for c in header if c not in {date_col, tgt_col}]
    return date_col, tgt_col, exog_cols


def _parse_iso_date(s: str) -> Optional[date]:
    try:
        # Ambil 10 char pertama (YYYY-MM-DD) untuk jaga-jaga
        return datetime.fromisoformat(s.strip()[:10]).date()
    except Exception:
        return None


def _infer_freq_from_dates(dates: List[date]) -> str:
    """
    Heuristik sederhana: lihat median delta (hari).
    - ~1  -> "D"
    - ~7  -> "W"
    - >=25 -> "M" (kasar; bulan bisa 28-31)
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


def read_train_meta(force_reload: bool = False) -> Dict[str, Any]:
    """
    Baca TRAIN_CSV untuk meta:
    - date_col, target_col, exog_columns
    - train_range (start, end)
    - freq (infer dari data atau DEFAULT_FREQ)
    Juga coba ambil order / seasonal_order dari model jika tersedia.
    """
    global _meta_cache
    if _meta_cache is not None and not force_reload:
        return _meta_cache

    meta: Dict[str, Any] = {
        "date_col": "Date",
        "target_col": "Revenue",
        "exog_columns": [],
        "train_range": {"start": None, "end": None},
        "freq": DEFAULT_FREQ or "D",
        "model_order": None,
        "model_seasonal_order": None,
        "exog_names_from_model": None,
    }

    # 1) Baca CSV untuk kolom & rentang tanggal
    if os.path.exists(TRAIN_CSV):
        with open(TRAIN_CSV, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            try:
                header = next(r)
            except StopIteration:
                header = []
            date_col, tgt_col, exog_cols = _detect_cols(header)
            meta["date_col"] = date_col
            meta["target_col"] = tgt_col

            # indeks kolom tanggal
            try:
                date_idx = header.index(date_col)
            except ValueError:
                date_idx = 0

            # kumpulkan tanggal utk infer freq & range
            dates: List[date] = []
            for row in r:
                if not row or len(row) <= date_idx:
                    continue
                d = _parse_iso_date(row[date_idx])
                if d:
                    dates.append(d)

            if dates:
                meta["train_range"] = {
                    "start": min(dates).isoformat(),
                    "end": max(dates).isoformat(),
                }

                # infer exog: buang kol target + tanggal
                meta["exog_columns"] = [c for c in header if c not in {date_col, tgt_col}]

                # coba infer freq dari tanggal
                inf = _infer_freq_from_dates(dates)
                meta["freq"] = inf or meta["freq"]

    # 2) Info dari model (order/seasonal/exog_names)
    m = load_model()
    if m is not None:
        try:
            # SARIMAXResults.model.order / seasonal_order
            order = getattr(m.model, "order", None)
            seas_order = getattr(m.model, "seasonal_order", None)
            meta["model_order"] = tuple(order) if order else None
            meta["model_seasonal_order"] = tuple(seas_order) if seas_order else None
        except Exception:
            pass
        try:
            # jika model tersimpan dengan exog_names
            exog_names = getattr(m.model, "exog_names", None)
            if exog_names and isinstance(exog_names, (list, tuple)):
                meta["exog_names_from_model"] = list(exog_names)
                # sinkronkan exog_columns bila kosong
                if not meta.get("exog_columns"):
                    meta["exog_columns"] = list(exog_names)
        except Exception:
            pass

    _meta_cache = meta
    return meta


# ============== FUTURE DATES ==============
def _add_months(d: date, n: int) -> date:
    """
    Tambah n bulan (kasar) ke tanggal d, menjaga day-of-month sebisanya.
    """
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    # clamp day
    day = min(d.day, _days_in_month(y, m))
    return date(y, m, day)


def _days_in_month(y: int, m: int) -> int:
    # sederhana, tanpa calendar lib
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m in (4, 6, 9, 11):
        return 30
    # February
    leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
    return 29 if leap else 28


def next_dates_from_train(h: int, freq: str = "D") -> List[str]:
    """
    Bangun daftar tanggal masa depan dengan panjang h.
    - D: hari+1, ... hari+h
    - W: minggu (kelipatan 7 hari)
    - M: tambah bulan
    Basis tanggal awal = last date dari train_range (atau hari ini jika tidak ada).
    """
    meta = read_train_meta()
    last = meta.get("train_range", {}).get("end")
    if last:
        start = datetime.fromisoformat(last).date()
    else:
        start = datetime.utcnow().date()

    f = (freq or meta.get("freq") or "D").upper()
    dates: List[str] = []

    if f == "W":
        for i in range(1, h + 1):
            d = start + timedelta(days=7 * i)
            dates.append(d.isoformat())
    elif f == "M":
        for i in range(1, h + 1):
            d = _add_months(start, i)
            dates.append(d.isoformat())
    else:
        # default D
        for i in range(1, h + 1):
            d = start + timedelta(days=i)
            dates.append(d.isoformat())

    return dates


# ============== FORECAST WRAPPER ==============
def _to_numpy_2d(ci_obj) -> np.ndarray:
    """
    Konversi conf_int ke np.ndarray shape (h, 2), baik jika sumbernya
    ndarray langsung atau objek DataFrame-like yang punya .to_numpy().
    """
    if isinstance(ci_obj, np.ndarray):
        arr = ci_obj
    else:
        # DataFrame-like
        try:
            arr = ci_obj.to_numpy()
        except Exception:
            # fallback: coba np.asarray
            arr = np.asarray(ci_obj)
    # pastikan 2 kolom (lower, upper)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2]
    # jika bentuknya aneh, buat NaN
    h = arr.shape[0] if arr.ndim >= 1 else 0
    out = np.full((h, 2), np.nan, dtype=float)
    return out


def forecast(h: int, alpha: float = 0.05,
             exog_future: Optional[List[List[float]]] = None
             ) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Jalankan get_forecast pada model yang di-load.
    Return:
      (mean, lower, upper) masing-masing list float dengan panjang h.
    """
    m = load_model()
    if m is None:
        return None

    try:
        res = m.get_forecast(steps=h, exog=exog_future)
        mean = res.predicted_mean
        # mean bisa berupa ndarray atau Series-like
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
        # Kalau exog tidak sesuai atau error lainnya, bubble up sebagai None
        # (Caller / endpoint akan mengembalikan 4xx/5xx dengan pesan yang jelas)
        return None
