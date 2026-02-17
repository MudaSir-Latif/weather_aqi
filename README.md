# 🌍 AQI Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Air Quality Index (AQI) Prediction System using Machine Learning and **Open-Meteo API** (no API key required!).

This system fetches real-time air quality data, performs feature engineering, trains ML models (Ridge Regression, Random Forest, XGBoost), and provides predictions through a REST API and interactive dashboard.

## 🌟 Features

- **No API Key Required**: Uses Open-Meteo API for free air quality data
- **Real-time AQI Monitoring**: Fetch current air quality data for any location
- **ML-based Predictions**: 3-day AQI forecasts using Ridge Regression, Random Forest, and XGBoost
- **Interactive Dashboard**: Beautiful Streamlit dashboard with visualizations
- **REST API**: FastAPI backend with comprehensive endpoints
- **Feature Engineering**: 22+ optimized features with temporal encoding and lag features
- **Historical Analysis**: Backfill and analyze historical air quality data
- **CI/CD Pipelines**: Automated data collection and model training

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [API Documentation](#api-documentation)
- [Dashboard Features](#dashboard-features)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Development](#development)
- [CI/CD](#cicd)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/MudaSir-Latif/AQI_prediction.git
cd AQI_prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your location coordinates
```

**Example `.env` configuration:**
```env
LATITUDE=24.8607
LONGITUDE=67.0011
CITY_NAME=Karachi

# Hopsworks Feature Store
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT_NAME=aqi_prediction
```

## 🎯 Quick Start

### 1. Fetch Historical Data

Backfill historical air quality data:
```bash
python scripts/backfill_data.py --days 365
```

### 2. Run Feature Pipeline

Engineer features from raw data:
```bash
python scripts/run_feature_pipeline.py --input data/raw/historical_data.csv
```

### 3. Train Models

Train ML models (Ridge, Random Forest, XGBoost):
```bash
python scripts/run_training_with_hopsworks.py
```

### 4. Validate Models

Validate trained models with recent data:
```bash
python scripts/validate_models.py --days 7
```

### 5. Start API Server

Launch the FastAPI backend:
```bash
cd api
python main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Launch Dashboard

Start the Streamlit dashboard:
```bash
cd dashboard
streamlit run app.py
# Dashboard available at http://localhost:8501
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Open-Meteo API                          │
│        (Air Quality & Weather Data - No API Key!)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Fetcher Module                        │
│  - Fetch air quality history                               │
│  - Fetch current air quality                               │
│  - Fetch weather forecast                                  │
│  - Calculate AQI (US EPA standard)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Module                     │
│  - Temporal features (cyclical encoding)                   │
│  - Lag features (1, 3, 6, 12, 24 hours)                   │
│  - Rolling statistics (mean, std, min, max)               │
│  - Pollutant interactions                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Model Training Module                        │
│  - Ridge Regression (primary model)                        │
│  - Random Forest                                           │
│  - XGBoost                                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   FastAPI API    │    │ Streamlit Dashboard│
│  - /api/current  │    │  - Real-time AQI │
│  - /api/predict  │    │  - Forecasts     │
│  - /api/historical│   │  - Visualizations│
│  - /api/model/info│   │  - Analytics     │
└──────────────────┘    └──────────────────┘
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "models_loaded": 3,
  "location": "Karachi (24.8607, 67.0011)"
}
```

#### 2. Current AQI
```http
GET /api/current
```

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "location": "Karachi",
  "latitude": 24.8607,
  "longitude": 67.0011,
  "aqi": 156,
  "aqi_category": "Unhealthy",
  "dominant_pollutant": "PM2.5",
  "pollutants": {
    "pm2_5": 65.2,
    "pm10": 120.5,
    "co": 450.3,
    "no2": 45.1,
    "so2": 12.3,
    "o3": 85.4
  }
}
```

#### 3. AQI Predictions
```http
GET /api/predict?days=3&model=ridge
```

**Query Parameters:**
- `days` (int, 1-7): Number of days to predict (default: 3)
- `model` (string): Model to use - "ridge", "random_forest", or "xgboost" (default: "ridge")

**Response:**
```json
[
  {
    "timestamp": "2024-01-02T12:00:00",
    "predicted_aqi": 142.5,
    "aqi_category": "Unhealthy for Sensitive Groups"
  },
  {
    "timestamp": "2024-01-03T12:00:00",
    "predicted_aqi": 138.2,
    "aqi_category": "Unhealthy for Sensitive Groups"
  },
  {
    "timestamp": "2024-01-04T12:00:00",
    "predicted_aqi": 145.8,
    "aqi_category": "Unhealthy for Sensitive Groups"
  }
]
```

#### 4. Historical Data
```http
GET /api/historical?days=7
```

**Query Parameters:**
- `days` (int, 1-365): Number of days of historical data (default: 7)

**Response:**
```json
[
  {
    "timestamp": "2024-01-01T12:00:00",
    "aqi": 156,
    "aqi_category": "Unhealthy",
    "pm2_5": 65.2,
    "pm10": 120.5,
    ...
  }
]
```

#### 5. Model Information
```http
GET /api/model/info
```

**Response:**
```json
[
  {
    "model_name": "Ridge",
    "mae": 12.45,
    "rmse": 18.32,
    "r2": 0.7234,
    "n_test_samples": 1825
  },
  {
    "model_name": "RandomForest",
    "mae": 10.82,
    "rmse": 15.67,
    "r2": 0.7891,
    "n_test_samples": 1825
  }
]
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

1. **Current Status Tab**
   - Real-time AQI gauge
   - Current pollutant concentrations
   - AQI category and health implications

2. **Historical Data Tab**
   - AQI trend over time
   - Statistical summary
   - Category distribution pie chart
   - Pollutant time series

3. **Predictions Tab**
   - 3-day AQI forecast
   - Predicted categories
   - Visualization of forecasts

4. **Model Info Tab**
   - Model performance metrics
   - Feature importance (Random Forest)
   - Success criteria

## 📈 Model Performance

### Success Criteria
- ✅ **MAE < 15**: Mean Absolute Error less than 15 AQI points
- ✅ **R² > 0.6**: Explains more than 60% of variance

### Typical Performance
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Ridge Regression | 12.5 | 18.3 | 0.72 |
| Random Forest | 10.8 | 15.7 | 0.79 |
| XGBoost | 11.2 | 16.4 | 0.77 |

### Top Features (by importance)
1. PM2.5 rolling mean (24h)
2. PM10 lag (1h)
3. Hour (cyclical)
4. PM2.5/PM10 ratio
5. NO2 rolling std (12h)

## ⚙️ Configuration

### Location Settings
Configure your location in `.env`:
```env
LATITUDE=24.8607      # Your latitude
LONGITUDE=67.0011     # Your longitude
CITY_NAME=Karachi     # Your city name
```

### Model Settings
```env
PREDICTION_DAYS=3              # Days to predict
MODEL_PATH=data/models         # Model storage path
LOOKBACK_HOURS=24              # Hours of lag features
ROLLING_WINDOW_SIZES=3,6,12,24 # Rolling window sizes
```

### Hopsworks Feature Store
```env
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT_NAME=aqi_prediction
```

## 🛠️ Development

### Project Structure
```
AQI_prediction/
├── api/
│   └── main.py              # FastAPI backend
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── src/
│   ├── config.py           # Configuration
│   ├── data_fetcher.py     # Open-Meteo integration
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   └── feature_store.py    # Hopsworks integration
├── scripts/
│   ├── backfill_data.py
│   ├── run_feature_pipeline.py
│   ├── run_training_with_hopsworks.py
│   └── validate_models.py
├── notebooks/
│   └── eda_analysis.py     # Exploratory analysis
├── data/
│   ├── raw/                # Raw data
│   ├── processed/          # Engineered features
│   └── models/             # Trained models
├── .github/workflows/
│   ├── feature_pipeline.yml
│   └── train_model.yml
├── requirements.txt
├── .env.example
└── README.md
```

### Running Tests
```bash
# Validate models
python scripts/validate_models.py

# Run EDA
python notebooks/eda_analysis.py --days 90
```

## 🔄 CI/CD

### Automated Workflows

1. **Feature Pipeline** (Daily at 6 AM UTC)
   - Fetches latest air quality data
   - Engineers features
   - Saves to feature store

2. **Model Training** (Weekly on Sunday at 2 AM UTC)
   - Backfills historical data
   - Trains all models
   - Validates performance
   - Saves model artifacts

### Manual Trigger
Go to Actions tab in GitHub and manually trigger workflows.

## 🌐 Open-Meteo API

This system uses **Open-Meteo API** which:
- ✅ Requires **no API key** (completely free!)
- ✅ Provides historical air quality data
- ✅ Includes weather forecasts
- ✅ Has generous rate limits
- ✅ Open-source and community-driven

**API Documentation:** https://open-meteo.com/en/docs/air-quality-api

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- **Open-Meteo** for providing free air quality data
- Original inspiration from [Talha-Bin-Sajid/AQI_Prediction_System](https://github.com/Talha-Bin-Sajid/AQI_Prediction_System)
- US EPA for AQI calculation standards

---

**Made with ❤️ for cleaner air**