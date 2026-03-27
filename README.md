# 💧 AquaAlert AI — Flood Detection & Early Warning Intelligence

> An end-to-end AI-powered flood detection and early warning system combining XGBoost, LSTM, CNN U-Net, SHAP Explainability, and GradCAM visualization — served via FastAPI with a React dashboard.



## 🚀 Features

| Feature | Description |
|---|---|
| **Flood Risk Prediction** | XGBoost classifier with real-time topographic telemetry sliders |
| **SHAP Explainability** | Per-prediction feature importance bar chart (XAI) |
| **Rainfall Prediction** | Kaggle S5E3 XGBoost model for atmospheric rainfall forecasting |
| **CNN Satellite Vision** | U-Net CNN segmentation on satellite imagery — detects flooded regions |
| **GradCAM Heatmap** | Gradient-weighted Class Activation Map overlaid on uploaded images |
| **Analytics Tab** | All training visualizations (EDA + Clustering) with lightbox viewer |
| **Notebook Gallery** | 6 Jupyter notebooks documented and linked |
| **Light / Dark Theme** | Full theme toggle, defaults to light mode |

---

## 🏗️ Architecture

```
flood-ai-system/
├── backend/                  # FastAPI server (Python 3.12)
│   ├── app.py                # Main FastAPI app
│   ├── routes/
│   │   ├── predict.py        # XGBoost + SHAP flood prediction
│   │   ├── vision.py         # CNN inference + GradCAM
│   │   ├── kaggle_rain.py    # Rainfall prediction
│   │   ├── rainfall.py       # Rainfall data route
│   │   └── plots.py          # Serve training visualizations
│   └── start_backend.bat     # Windows launcher
│
├── frontend/                 # React + Vite dashboard
│   └── src/
│       ├── App.jsx           # Main dashboard component
│       └── index.css         # Design system (light + dark themes)
│
├── models/saved/             # Trained model files
│   ├── xgboost_classifier.pkl
│   ├── cnn_model.h5
│   ├── lstm_model.h5
│   ├── classifier.pkl
│   ├── kaggle_rainfall_model.pkl
│   └── kaggle_scaler.pkl
│
├── notebooks/                # Jupyter training notebooks
│   ├── data_analysis.ipynb
│   ├── clustering.ipynb
│   ├── cnn_satellite.ipynb
│   ├── lstm_model.ipynb
│   ├── lstm_classifier.ipynb
│   └── kaggle_s5e3_predictor.ipynb
│
├── data/
│   ├── raw/flood-area/       # Satellite flood imagery (Image/ + Mask/)
│   └── processed/            # Cleaned & clustered datasets
│
└── outputs/plots/            # Generated training visualizations
    ├── plot1_heatmap.png
    ├── plot2_distribution.png
    └── clustering/
```

---

## 🤖 Models

### 1. XGBoost Flood Classifier
- **Input:** Rainfall, Temperature, Humidity, River Discharge, Water Level, Elevation, Risk Zone
- **Output:** Binary flood prediction + probability
- **XAI:** SHAP TreeExplainer — per-feature contribution chart

### 2. CNN U-Net Segmentation
- **Architecture:** Encoder-Decoder with UpSampling2D
- **Input:** 128×128 RGB satellite image
- **Output:** Binary flood mask + flood percentage
- **XAI:** GradCAM heatmap (jet colormap overlay)

### 3. LSTM Time-Series
- **Input:** Sequential meteorological data
- **Output:** Flood probability over time

### 4. Kaggle S5E3 XGBoost
- **Input:** Pressure, Temperature, Humidity, Cloud Cover, Wind Speed, Sunshine
- **Output:** Rainfall probability (RAINING / CLEAR)

### 5. KMeans Clustering
- **Purpose:** Spatial risk zone segmentation (Zone 0=Safe, 1=Moderate, 2=Danger)
- **Used as:** Feature input to XGBoost classifier

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.12
- Node.js 18+

### Backend

```bash
cd backend

# Install dependencies
C:\Users\...\Python312\python.exe -m pip install fastapi uvicorn tensorflow pillow numpy requests pydantic scikit-learn xgboost joblib pandas shap

# Start server
start_backend.bat
# OR
python app.py
```

Backend runs at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Flood risk prediction + SHAP values |
| `POST` | `/api/v1/vision` | CNN inference + GradCAM heatmap |
| `POST` | `/api/v1/kaggle/predict` | Rainfall prediction |
| `GET` | `/api/v1/plots` | List all training visualizations |
| `GET` | `/api/v1/plots/image?file=...` | Serve a plot image |

### Example: Flood Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"rainfall":340,"temperature":28,"humidity":85,"river_discharge":1060,"water_level":7.9,"elevation":126,"risk_zone":1}'
```

```json
{
  "algorithm": "XGBoost + XAI",
  "prediction_status": "DANGER: IMMINENT FLOOD",
  "flood_probability": 58.72,
  "recommended_action": "Evacuate immediately",
  "shap_values": [
    {"feature": "Rainfall", "value": 0.45, "pct": 51.0, "direction": "positive"},
    ...
  ]
}
```

---

## 🧠 Explainable AI (XAI)

### SHAP (XGBoost)
Every flood prediction includes SHAP values showing which features drove the decision:
- **Positive (blue)** → increases flood risk
- **Negative (grey)** → decreases flood risk

### GradCAM (CNN)
After satellite image inference, a GradCAM heatmap is overlaid:
- **Red** → high flood activation regions
- **Blue** → low activation regions

---

## 📊 Training Data

- **Source:** Synthetic Indian subcontinent flood dataset + Kaggle S5E3
- **Samples:** ~1,000 labeled flood/no-flood records
- **Satellite Images:** 290 flood-area images with binary masks (128×128)
- **Features:** Latitude, Longitude, Rainfall, Temperature, Humidity, River Discharge, Water Level, Elevation, Land Cover, Soil Type, Population Density, Infrastructure, Historical Floods

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Lucide Icons, Axios |
| Backend | FastAPI, Uvicorn, Python 3.12 |
| ML Models | XGBoost, TensorFlow/Keras, Scikit-learn |
| XAI | SHAP, GradCAM (custom TF implementation) |
| Data | Pandas, NumPy, Pillow |
| Notebooks | Jupyter Lab |

---

## 📁 Notebooks

| Notebook | Description |
|---|---|
| `data_analysis.ipynb` | EDA — heatmaps, distributions, outlier detection |
| `clustering.ipynb` | KMeans spatial risk zone segmentation |
| `cnn_satellite.ipynb` | U-Net CNN training on flood satellite imagery |
| `lstm_model.ipynb` | LSTM time-series flood prediction |
| `lstm_classifier.ipynb` | LSTM binary flood classifier |
| `kaggle_s5e3_predictor.ipynb` | XGBoost rainfall prediction (Kaggle S5E3) |

---

## 🔮 Future Improvements

- [ ] Real-time NASA POWER API integration for live weather data
- [ ] GeoJSON flood zone map overlay
- [ ] Alert notification system (email/SMS)
- [ ] Docker containerization
- [ ] Deploy to AWS EC2 / Azure

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ — AquaAlert AI Flood Intelligence Platform*
