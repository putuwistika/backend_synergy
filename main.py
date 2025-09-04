import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# === Settings dari ENV ===
TZ = os.getenv("TZ", "Asia/Jakarta")
DEFAULT_FREQ = os.getenv("DEFAULT_FREQ", "D")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/sarimax_model.pkl")
TRAIN_CSV = os.getenv("TRAIN_CSV", "artifacts/train_df.csv")
TEST_CSV = os.getenv("TEST_CSV", "artifacts/test_df.csv")
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",")]

app = FastAPI(title="Forecast Backend (SARIMAX)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == ["*"] else ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    """Liveness probe untuk Render."""
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    """
    Readiness probe: cek apakah artefak tersedia.
    Belum load model dulu—cukup cek keberadaan file.
    """
    missing = []
    for p in [MODEL_PATH, TRAIN_CSV, TEST_CSV]:
        if not os.path.exists(p):
            missing.append(p)

    return {
        "ready": len(missing) == 0,
        "missing": missing,
        "env": {
            "TZ": TZ,
            "DEFAULT_FREQ": DEFAULT_FREQ,
            "MODEL_PATH": MODEL_PATH,
            "TRAIN_CSV": TRAIN_CSV,
            "TEST_CSV": TEST_CSV,
        },
    }

@app.get("/meta")
def meta_minimal():
    """
    Meta minimal—nanti akan kita ganti agar membaca detail model (order, seasonal_order, exog cols, dst.)
    Sekarang cukup echo path & default freq supaya FE bisa render awal.
    """
    return {
        "model_name": "sarimax",
        "default_freq": DEFAULT_FREQ,
        "artifacts": {
            "model_path": MODEL_PATH,
            "train_csv": TRAIN_CSV,
            "test_csv": TEST_CSV,
            "exists": {
                "model": os.path.exists(MODEL_PATH),
                "train": os.path.exists(TRAIN_CSV),
                "test": os.path.exists(TEST_CSV),
            },
        },
    }
