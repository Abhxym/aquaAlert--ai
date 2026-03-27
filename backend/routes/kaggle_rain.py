from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

router = APIRouter()

models_dir = os.path.join("..", "models", "saved")
model_path = os.path.join(models_dir, "kaggle_rainfall_model.pkl")
scaler_path = os.path.join(models_dir, "kaggle_scaler.pkl")

xgb_model = None
scaler = None

try:
    if os.path.exists(model_path):
        xgb_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading Kaggle Model: {e}")

class KaggleParams(BaseModel):
    pressure: float = 1015.0
    maxtemp: float = 28.0
    temparature: float = 24.0
    mintemp: float = 20.0
    dewpoint: float = 18.0
    humidity: float = 85.0
    cloud: float = 90.0
    sunshine: float = 1.5
    winddirection: float = 180.0
    windspeed: float = 20.0

@router.post("/predict")
async def predict_rainfall(params: KaggleParams):
    if xgb_model is None or scaler is None:
        return {"error": "Kaggle Environment Missing! Run 'kaggle_s5e3_predictor.ipynb' inside VS Code to serialize the models!"}
    
    # Replicating the exact tabular sequence from Kaggle train.csv 
    data = {
        'pressure': [params.pressure],
        'maxtemp': [params.maxtemp],
        'temparature': [params.temparature],
        'mintemp': [params.mintemp],
        'dewpoint': [params.dewpoint],
        'humidity': [params.humidity],
        'cloud': [params.cloud],
        'sunshine': [params.sunshine],
        'winddirection': [params.winddirection],
        'windspeed': [params.windspeed]
    }
    
    df = pd.DataFrame(data)
    
    try:
        X_scaled = scaler.transform(df)
        pred = xgb_model.predict(X_scaled)[0]
        prob = float(xgb_model.predict_proba(X_scaled)[0][1] * 100)
    except Exception as e:
        return {"error": str(e)}
        
    return {
        "status": "RAINING" if pred == 1 else "CLEAR",
        "rainfall_probability": prob,
        "recommended_action": "S5E3 Kaggle Dataset Confirms Storm Cells. Inject this predicted Rainfall output downstream into the primary Flood Topography Matrix!" if prob > 60 else "No substantial meteorological anomalies detected."
    }
