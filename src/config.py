"""
Configuration management for AQI Prediction System
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class OpenMeteoConfig:
    """Configuration for Open-Meteo API"""
    
    air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url: str = "https://api.open-meteo.com/v1/forecast"
    timeout: int = int(os.getenv("OPENMETEO_TIMEOUT", 120))
    
    # Air quality parameters to fetch
    air_quality_params: list = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "aerosol_optical_depth",
        "dust",
        "ammonia"
    ]
    
    # Weather parameters to fetch
    weather_params: list = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
        "pressure_msl"
    ]


class LocationConfig:
    """Location configuration"""
    
    latitude: float = float(os.getenv("LATITUDE", 24.8607))
    longitude: float = float(os.getenv("LONGITUDE", 67.0011))
    city_name: str = os.getenv("CITY_NAME", "Karachi")


class HopsworksConfig:
    """Configuration for Hopsworks Feature Store"""
    
    api_key: Optional[str] = os.getenv("HOPSWORKS_API_KEY")
    project_name: str = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_prediction")
    feature_group_name: str = "aqi_features"
    feature_group_version: int = 1


class ModelConfig:
    """Model training and prediction configuration"""
    
    prediction_days: int = int(os.getenv("PREDICTION_DAYS", 3))
    model_path: Path = Path(os.getenv("MODEL_PATH", "data/models"))
    
    # Feature engineering
    lookback_hours: int = int(os.getenv("LOOKBACK_HOURS", 24))
    rolling_window_sizes: list = [int(x) for x in os.getenv("ROLLING_WINDOW_SIZES", "3,6,12,24").split(",")]
    
    # Model parameters
    models_to_train: list = ["ridge", "random_forest", "xgboost", "lstm"]
    test_size: float = 0.2
    random_state: int = 42

    # LSTM parameters
    lstm_epochs: int = int(os.getenv("LSTM_EPOCHS", 50))
    lstm_batch_size: int = int(os.getenv("LSTM_BATCH_SIZE", 32))
    lstm_hidden_size: int = int(os.getenv("LSTM_HIDDEN_SIZE", 64))
    lstm_lookback: int = int(os.getenv("LSTM_LOOKBACK", 24))


class AlertConfig:
    """Configuration for AQI hazard alerts"""

    # AQI thresholds (US EPA standard)
    thresholds: dict = {
        "good": {"min": 0, "max": 50, "level": "info"},
        "moderate": {"min": 51, "max": 100, "level": "info"},
        "unhealthy_sensitive": {"min": 101, "max": 150, "level": "warning"},
        "unhealthy": {"min": 151, "max": 200, "level": "alert"},
        "very_unhealthy": {"min": 201, "max": 300, "level": "alert"},
        "hazardous": {"min": 301, "max": 500, "level": "critical"},
    }

    # Webhook / notification settings (optional)
    webhook_url: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")
    alert_email: Optional[str] = os.getenv("ALERT_EMAIL")
    alert_log_path: Path = Path("logs/aqi_alerts.json")

    # Cooldown: minimum seconds between repeated alerts of the same level
    cooldown_seconds: int = int(os.getenv("ALERT_COOLDOWN", 3600))


class AppConfig:
    """Application configuration"""
    
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Ensure model directory exists
    ModelConfig.model_path.mkdir(parents=True, exist_ok=True)


# Configure logger
logger.add(
    "logs/aqi_prediction.log",
    rotation="1 day",
    retention="7 days",
    level=AppConfig.log_level
)

logger.info("Configuration loaded successfully")
