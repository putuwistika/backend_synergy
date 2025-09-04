from fastapi import FastAPI
import os, os.path as op

DEFAULT_FREQ = os.getenv("DEFAULT_FREQ", "D")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/sarimax_model.pkl")
TRAIN_CSV  = os.getenv("TRAIN_CSV",  "artifacts/train_df.csv")
TEST_CSV   = os.getenv("TEST_CSV",   "artifacts/test_df.csv")

app = FastAPI(title="Forecast Backend — Vercel", version="0.1.0")

@app.get("/api/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/api/readyz")
def readyz():
    missing = [p for p in [MODEL_PATH, TRAIN_CSV, TEST_CSV] if not op.exists(p)]
    return {"ready": len(missing) == 0, "missing": missing}

@app.get("/api/meta")
def meta():
    return {
        "model_name": "sarimax",
        "default_freq": DEFAULT_FREQ,
        "artifacts": {
            "model_path": MODEL_PATH,
            "train_csv": TRAIN_CSV,
            "test_csv": TEST_CSV,
            "exists": {
                "model": op.exists(MODEL_PATH),
                "train": op.exists(TRAIN_CSV),
                "test": op.exists(TEST_CSV),
            },
        },
    }
