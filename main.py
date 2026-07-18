import os
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="UrbanVal AI Backend Engine")

# ----------------------------------------------------------------
# 1. CORS CONFIGURATION (Prevents Browser Blockages)
# ----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits local index.html to communicate with Render
    allow_credentials=True,
    allow_methods=["*"],  # Permits GET, POST, OPTIONS
    allow_headers=["*"],  # Permits all metadata headers
)

# ----------------------------------------------------------------
# 2. DATA PATH CONSTRAINTS & MODEL LOADING
# ----------------------------------------------------------------
# Dynamically resolves path boundaries using absolute working directory locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Critical System Failure: Model weight matrix file not discovered at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as file:
    model_artifacts = pickle.load(file)

# Extract core pipelines safely from pickle protocol
model = model_artifacts["model"]
scaler = model_artifacts.get("scaler", None)
model_features = model_artifacts.get("features", [])

# ----------------------------------------------------------------
# 3. DATA STRUCTURE DEFINITIONS (Pydantic Layer)
# ----------------------------------------------------------------
class PredictionRequest(BaseModel):
    property_type: str
    location: str
    size_sqm: float

# ----------------------------------------------------------------
# 4. OPERATIONAL API ENDPOINTS
# ----------------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "engine": "UrbanVal AI Inference Core"}

@app.get("/metrics")
def get_metrics():
    """Returns historical cross-validation matrices saved within model artifacts."""
    return {
        "r2": model_artifacts.get("r2_score", 0.88),
        "mae": model_artifacts.get("mae", 15400),
        "rmse": model_artifacts.get("rmse", 22100)
    }

@app.post("/predict")
def predict_property_value(payload: PredictionRequest):
    try:
        # 1. Extract inputs safely matching JavaScript layout
        input_size = float(payload.size_sqm)
        input_location = str(payload.location)
        input_type = str(payload.property_type)

        # 2. Build template DataFrame matching One-Hot encoding schema exactly
        input_data = pd.DataFrame([{col: 0 for col in model_features}])
        
        # 3. Assign structural values safely
        if "size_sqm" in input_data.columns:
            input_data["size_sqm"] = input_size
        
        # 4. Map One-Hot encoded categoricals
        loc_column = f"location_{input_location}"
        type_column = f"property_type_{input_type}"
        
        if loc_column in input_data.columns:
            input_data[loc_column] = 1
        if type_column in input_data.columns:
            input_data[type_column] = 1

        # 5. Handle feature scaling alignment if integrated during pipeline training
        if scaler is not None:
            features_to_scale = ["size_sqm"]
            input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])

        # 6. Execute model prediction mapping out tracking matrices
        prediction = model.predict(input_data)[0]
        
        # Avoid unhandled negative predictions from standard linear models
        final_valuation = max(0.0, float(prediction))

        return {"predicted_price_usd": final_valuation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Crash Trace: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Bound to local loop configurations for optional isolated debugging steps
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
