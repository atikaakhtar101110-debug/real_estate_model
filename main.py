import os
import time
import threading
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# UrbanVal AI - Production Backend
# Compatible with:
# ✔ Hugging Face Spaces
# ✔ Render
# ✔ Localhost
# ============================================================

app = FastAPI(
    title="UrbanVal AI Backend",
    version="2.0.0",
    description="Real Estate Price Prediction API"
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load ML Model
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "real_estate_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Unable to load model: {e}")

# ============================================================
# Detect Feature Names
# ============================================================

DEFAULT_FEATURES = [
    "size_sqm",
    "location_New York",
    "location_Miami",
    "location_Los Angeles",
    "location_Chicago",
    "location_Houston",
    "property_type_Villa",
    "property_type_Retail",
    "property_type_Warehouse",
    "property_type_Apartment",
    "property_type_Office"
]

try:
    if hasattr(model, "feature_names_in_"):
        MODEL_FEATURES = list(model.feature_names_in_)
    else:
        MODEL_FEATURES = DEFAULT_FEATURES
except Exception:
    MODEL_FEATURES = DEFAULT_FEATURES

# ============================================================
# Live Statistics
# ============================================================

stats_lock = threading.Lock()

live_stats = {
    "total_predictions": 0,
    "average_prediction": 0.0,
    "highest_prediction": 0.0,
    "lowest_prediction": None,
    "last_prediction": 0.0,
    "response_time_ms": 0.0
}

# ============================================================
# Model Evaluation Metrics
# (Replace with your actual values after training)
# ============================================================

MODEL_METRICS = {
    "r2": 0.88,
    "mae": 15400,
    "rmse": 22100
}

# ============================================================
# Request Model
# ============================================================

class PredictionRequest(BaseModel):
    property_type: str
    location: str
    size_sqm: float

# ============================================================
# Helper Function
# ============================================================

def update_live_stats(prediction: float, response_time: float):
    with stats_lock:
        live_stats["total_predictions"] += 1
        count = live_stats["total_predictions"]
        previous_average = live_stats["average_prediction"]

        live_stats["average_prediction"] = (
            (previous_average * (count - 1)) + prediction
        ) / count

        live_stats["last_prediction"] = prediction
        live_stats["response_time_ms"] = round(response_time, 2)

        if prediction > live_stats["highest_prediction"]:
            live_stats["highest_prediction"] = prediction

        if (
            live_stats["lowest_prediction"] is None
            or prediction < live_stats["lowest_prediction"]
        ):
            live_stats["lowest_prediction"] = prediction


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "UrbanVal AI Backend",
        "version": "2.0.0",
        "message": "API is running successfully."
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# Model Metrics
# ============================================================

@app.get("/metrics")
def metrics():
    return {
        "model_metrics": MODEL_METRICS,
        "live_metrics": live_stats
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(payload: PredictionRequest):
    start_time = time.perf_counter()

    try:
        # -----------------------------
        # Create empty feature dataframe
        # -----------------------------
        input_df = pd.DataFrame(
            [{feature: 0 for feature in MODEL_FEATURES}]
        )

        # Numerical feature
        if "size_sqm" in input_df.columns:
            input_df.loc[0, "size_sqm"] = float(payload.size_sqm)

        # One-hot encoding
        location_column = f"location_{payload.location}"
        property_column = f"property_type_{payload.property_type}"

        if location_column in input_df.columns:
            input_df.loc[0, location_column] = 1

        if property_column in input_df.columns:
            input_df.loc[0, property_column] = 1

        # Prediction
        prediction = model.predict(input_df)[0]
        prediction = max(0.0, float(prediction))

        # Response time
        elapsed = (time.perf_counter() - start_time) * 1000

        # Update live statistics
        update_live_stats(prediction, elapsed)

        # Return response
        return {
            "success": True,
            "predicted_price_usd": round(prediction, 2),
            "model_metrics": MODEL_METRICS,
            "live_metrics": live_stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": str(e)
            }
        )


# ============================================================
# Run Locally
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
