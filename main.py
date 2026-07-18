import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Initialize FastAPI app configuration
app = FastAPI(
    title="Real Estate Price Prediction API",
    description="A production-ready API for predicting property values.",
    version="1.0.0"
)

# 2. BIND CORS MIDDLEWARE FIRST (Crucial for Hugging Face Web Security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all client web applications to safely query the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the expected incoming JSON data structure using Pydantic
class PropertyFeatures(BaseModel):
    PropertyType: str  # Expecting: 'Villa', 'Apartment', 'Office', 'Retail', 'Warehouse'
    Location: str      # Expecting: 'New York', 'Los Angeles', 'Miami', 'Chicago', 'Houston'
    Size_sqm: float    # Expecting: numerical value e.g., 120.5

# 4. Safely load your pre-existing model pipeline (Left completely intact)
MODEL_PATH = 'real_estate_model.joblib'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("🚀 Success: Model pipeline successfully initialized.")
else:
    model = None
    print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")

# 5. Root verification endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Real Estate Price Prediction API. Go to /docs for the UI."}

# 6. Prediction evaluation endpoint
@app.post("/predict")
def predict_price(features: PropertyFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded/available on server.")

    try:
        # Convert incoming JSON data into a DataFrame matching the model's training shape
        input_df = pd.DataFrame([features.model_dump()])

        # Generate the numerical array prediction response
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "predicted_price_usd": round(float(prediction), 2),
            "input_summary": features.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
