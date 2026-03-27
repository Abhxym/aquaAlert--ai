from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, "models", "saved", "xgboost_classifier.pkl")

xgb_model = None
FEATURE_COLS = []

try:
    xgb_model = joblib.load(model_path)
    FEATURE_COLS = list(xgb_model.feature_names_in_)
    print(f"[predict] XGBoost loaded OK. Cols: {FEATURE_COLS}")
except Exception as e:
    print(f"[predict] Load failed: {e}")

DISPLAY_LABELS = ['Rainfall', 'Temperature', 'Humidity',
                  'River Discharge', 'Water Level', 'Elevation', 'Risk Zone']


class PredictionRequest(BaseModel):
    rainfall: float
    temperature: float
    humidity: float
    river_discharge: float
    water_level: float
    elevation: float
    risk_zone: int


@router.post("/predict")
def predict_flood(req: PredictionRequest):
    if xgb_model is None or not FEATURE_COLS:
        return {
            "algorithm": "XGBoost",
            "prediction_status": "API OFFLINE",
            "flood_probability": 0.0,
            "recommended_action": "Backend model not loaded",
            "shap_values": [],
        }

    values = [
        req.rainfall, req.temperature, req.humidity,
        req.river_discharge, req.water_level, req.elevation, req.risk_zone
    ]
    df = pd.DataFrame([values], columns=FEATURE_COLS)

    try:
        pred = int(xgb_model.predict(df)[0])
    except Exception as e:
        return {"prediction_status": "API OFFLINE", "flood_probability": 0.0,
                "recommended_action": str(e), "shap_values": []}

    try:
        probability = float(xgb_model.predict_proba(df)[0][1] * 100)
    except Exception:
        probability = 99.0 if pred == 1 else 1.0

    # XAI: try SHAP first, fall back to feature importances
    shap_values = []
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(df)
        vals = np.array(sv).flatten()[:len(FEATURE_COLS)]
        total = float(np.sum(np.abs(vals))) or 1.0
        shap_values = sorted([
            {
                "feature": DISPLAY_LABELS[i],
                "value": round(float(vals[i]), 4),
                "pct": round(abs(float(vals[i])) / total * 100, 1),
                "direction": "positive" if float(vals[i]) > 0 else "negative",
            }
            for i in range(len(FEATURE_COLS))
        ], key=lambda x: abs(x["value"]), reverse=True)
    except Exception:
        try:
            fi = xgb_model.feature_importances_
            total = float(np.sum(fi)) or 1.0
            shap_values = sorted([
                {
                    "feature": DISPLAY_LABELS[i],
                    "value": round(float(fi[i]), 4),
                    "pct": round(float(fi[i]) / total * 100, 1),
                    "direction": "positive",
                }
                for i in range(len(FEATURE_COLS))
            ], key=lambda x: x["value"], reverse=True)
        except Exception:
            shap_values = []

    return {
        "algorithm": "XGBoost + XAI",
        "prediction_status": "DANGER: IMMINENT FLOOD" if pred == 1 else "SAFE",
        "flood_probability": round(probability, 2),
        "recommended_action": "Evacuate immediately" if pred == 1 else "Monitor conditions",
        "shap_values": shap_values,
    }
