from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import predict, rainfall, vision, kaggle_rain, plots
import uvicorn

app = FastAPI(
    title="Urban Flood Detection System API",
    description="AI powered real-time disaster tracking utilizing XGBoost, LSTM, and NASA CNN Vision.",
    version="1.0.0"
)

# Securing the frontend bridge
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Binding the functional sub-routes dynamically
app.include_router(predict.router, prefix="/api/v1")
app.include_router(rainfall.router, prefix="/api/v1")
app.include_router(vision.router, prefix="/api/v1")
app.include_router(kaggle_rain.router, prefix="/api/v1/kaggle")
app.include_router(plots.router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {
        "status": "Secure Database Active",
        "description": "Urban Flood AI Engine Listening on Port 8000"
    }

if __name__ == "__main__":
    # Natively supports concurrent asyncio loading for instant ML responses
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
