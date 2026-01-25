"""
FastAPI backend for AQI Prediction System
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import OpenMeteoFetcher, AQICalculator
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.config import LocationConfig, ModelConfig

# Initialize FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="Air Quality Index Prediction System using Open-Meteo API and Machine Learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
fetcher = OpenMeteoFetcher()
engineer = FeatureEngineer()
trainer = ModelTrainer()
location = LocationConfig()
config = ModelConfig()

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load trained models on startup"""
    try:
        logger.info("Loading trained models...")
        model_dir = config.model_path
        
        if model_dir.exists():
            trainer.load_models(model_dir)
            logger.info(f"Loaded {len(trainer.models)} models")
        else:
            logger.warning(f"Model directory not found: {model_dir}")
            logger.warning("Predictions will not be available until models are trained")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


# Response models
class CurrentAQIResponse(BaseModel):
    """Current AQI data response"""
    timestamp: str
    location: str
    latitude: float
    longitude: float
    aqi: int
    aqi_category: str
    dominant_pollutant: str
    pollutants: Dict[str, Optional[float]]


class PredictionResponse(BaseModel):
    """AQI prediction response"""
    timestamp: str
    predicted_aqi: float
    aqi_category: str
    confidence_interval: Optional[Dict[str, float]] = None


class HistoricalDataPoint(BaseModel):
    """Historical data point"""
    timestamp: str
    aqi: int
    aqi_category: str
    pm2_5: Optional[float]
    pm10: Optional[float]
    co: Optional[float]
    no2: Optional[float]
    so2: Optional[float]
    o3: Optional[float]


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str
    mae: float
    rmse: float
    r2: float
    n_test_samples: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    location: str


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(trainer.models),
        "location": f"{location.city_name} ({location.latitude}, {location.longitude})"
    }


@app.get("/api/current", response_model=CurrentAQIResponse)
async def get_current_aqi():
    """Get current AQI and pollutant data"""
    try:
        current_data = fetcher.fetch_current_air_quality()
        
        if not current_data:
            raise HTTPException(status_code=503, detail="Failed to fetch current air quality data")
        
        return CurrentAQIResponse(
            timestamp=current_data['timestamp'],
            location=current_data['location'],
            latitude=current_data['latitude'],
            longitude=current_data['longitude'],
            aqi=current_data['aqi'],
            aqi_category=current_data['aqi_category'],
            dominant_pollutant=current_data['dominant_pollutant'],
            pollutants={
                'pm2_5': current_data.get('pm2_5'),
                'pm10': current_data.get('pm10'),
                'co': current_data.get('co'),
                'no2': current_data.get('no2'),
                'so2': current_data.get('so2'),
                'o3': current_data.get('o3'),
                'nh3': current_data.get('nh3'),
                'dust': current_data.get('dust')
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_aqi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict", response_model=List[PredictionResponse])
async def get_predictions(
    days: int = Query(default=3, ge=1, le=7, description="Number of days to predict"),
    model: str = Query(default="ridge", description="Model to use (ridge, random_forest, xgboost)")
):
    """Get AQI predictions for upcoming days"""
    try:
        if not trainer.models:
            raise HTTPException(
                status_code=503, 
                detail="Models not loaded. Please train models first."
            )
        
        if model not in trainer.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' not found. Available models: {list(trainer.models.keys())}"
            )
        
        # Fetch recent data for feature engineering
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Get past week for lag features
        
        df = fetcher.fetch_air_quality_history(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if df.empty:
            raise HTTPException(status_code=503, detail="Failed to fetch historical data for predictions")
        
        # Engineer features
        df_engineered = engineer.engineer_features(df)
        
        # Get the most recent complete record for prediction
        X, _, _ = engineer.prepare_training_data(df_engineered)
        
        if X.empty:
            raise HTTPException(status_code=500, detail="Failed to prepare features for prediction")
        
        # Use the most recent row as base for predictions
        X_latest = X.iloc[-1:].copy()
        
        predictions = []
        aqi_calc = AQICalculator()
        
        for day in range(1, days + 1):
            # Make prediction
            pred_aqi = trainer.predict(X_latest, model_name=model)[0]
            pred_aqi = max(0, min(500, pred_aqi))  # Clamp to valid AQI range
            
            pred_timestamp = (end_date + timedelta(days=day)).isoformat()
            pred_category = aqi_calc.get_aqi_category(int(pred_aqi))
            
            predictions.append(PredictionResponse(
                timestamp=pred_timestamp,
                predicted_aqi=round(float(pred_aqi), 2),
                aqi_category=pred_category
            ))
        
        return predictions
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical", response_model=List[HistoricalDataPoint])
async def get_historical(
    days: int = Query(default=7, ge=1, le=365, description="Number of days of historical data")
):
    """Get historical AQI time-series data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = fetcher.fetch_air_quality_history(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if df.empty:
            raise HTTPException(status_code=503, detail="Failed to fetch historical data")
        
        # Convert to response format
        historical_data = []
        for _, row in df.iterrows():
            historical_data.append(HistoricalDataPoint(
                timestamp=row['time'].isoformat(),
                aqi=int(row.get('aqi', 0)),
                aqi_category=row.get('aqi_category', 'Unknown'),
                pm2_5=row.get('pm2_5'),
                pm10=row.get('pm10'),
                co=row.get('co'),
                no2=row.get('no2'),
                so2=row.get('so2'),
                o3=row.get('o3')
            ))
        
        return historical_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_historical: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info", response_model=List[ModelMetrics])
async def get_model_info():
    """Get model performance metrics"""
    try:
        if not trainer.metrics:
            raise HTTPException(
                status_code=503,
                detail="Model metrics not available. Please train models first."
            )
        
        metrics_list = []
        for model_name, metrics in trainer.metrics.items():
            metrics_list.append(ModelMetrics(
                model_name=metrics['model_name'],
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                r2=metrics['r2'],
                n_test_samples=metrics['n_test_samples']
            ))
        
        return metrics_list
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_model_info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AQI Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
