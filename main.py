from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# 1. Initialize FastAPI app
app = FastAPI(
    title="Real Estate Price Prediction API",
    description="A production-ready API for predicting property values.",
    version="1.0.0"
)

# 2. Define the expected input data structure using Pydantic
class PropertyFeatures(BaseModel):
    PropertyType: str  # e.g., 'Villa', 'Apartment', 'Office', 'Retail', 'Warehouse'
    Location: str      # e.g., 'New York', 'Los Angeles', 'Miami', 'Chicago', 'Houston'
    Size_sqm: float    # e.g., 120.5

# 3. Safely load the model pipeline
MODEL_PATH = 'real_estate_model.joblib'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")

# 4. Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Real Estate Price Prediction API. Go to /docs for the UI."}

# 5. Prediction endpoint
@app.post("/predict")
def predict_price(features: PropertyFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded/available.")

    try:
        # Convert incoming JSON data into a DataFrame matching the model's training shape
        input_df = pd.DataFrame([features.model_dump()])

        # Generate prediction
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "predicted_price_usd": round(prediction, 2),
            "input_summary": features.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
