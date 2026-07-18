import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# 1. Initialize FastAPI app configuration
app = FastAPI(
    title="Real Estate Price Prediction API",
    description="A production-ready API for predicting property values with live validation metrics.",
    version="2.2.0"
)

# 2. CORS Middleware for Hugging Face Web Security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the expected incoming JSON data structure using Pydantic
class PropertyFeatures(BaseModel):
    PropertyType: str  
    Location: str      
    Size_sqm: float    

# 4. Global variables for model and computed metrics
MODEL_PATH = 'real_estate_model.joblib'
DATA_PATH = 'RealEstateAgencyData.xlsx - Properties.csv'

live_metrics = {
    "r2": "N/A",
    "mae": "N/A",
    "rmse": "N/A"
}
model = None

# 5. Initialization logic to load model and compute live metrics from your CSV
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("🚀 Success: Model pipeline successfully initialized.")
    
    if os.path.exists(DATA_PATH):
        try:
            test_df = pd.read_csv(DATA_PATH)
            
            # Explicitly select only the exact features the model expects
            feature_columns = ['PropertyType', 'Location', 'Size_sqm']
            target_column = 'PriceUSD'
            
            # Check if all required columns exist in the uploaded file
            if all(col in test_df.columns for col in feature_columns + [target_column]):
                X_test = test_df[feature_columns]
                y_true = test_df[target_column]
                
                # Run batch predictions across your dataset rows
                y_pred = model.predict(X_test)
                
                # Compute and round live statistical metrics
                live_metrics["r2"] = round(float(r2_score(y_true, y_pred)), 3)
                live_metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 2)
                live_metrics["rmse"] = round(float(root_mean_squared_error(y_true, y_pred)), 2)
                print("📈 Live performance metrics computed successfully from explicit columns.")
            else:
                missing = [c for c in feature_columns + [target_column] if c not in test_df.columns]
                print(f"⚠️ Warning: Missing required columns in CSV: {missing}")
        except Exception as e:
            print(f"⚠️ Error calculating metrics: {str(e)}")
    else:
        print(f"⚠️ Warning: Dataset file not found at {DATA_PATH}. Check your repository spelling.")
else:
    print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")

# 6. Root verification endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Real Estate Price Prediction API. Go to /docs for the UI."}

# 7. Endpoint to serve live statistics to your HTML interface
@app.get("/metrics")
def get_metrics():
    return live_metrics

# 8. Prediction evaluation endpoint
@app.post("/predict")
def predict_price(features: PropertyFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded/available on server.")

    try:
        input_df = pd.DataFrame([features.model_dump()])
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "predicted_price_usd": round(float(prediction), 2),
            "input_summary": features.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
