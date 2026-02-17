"""
Validate trained models
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_trainer import ModelTrainer
from src.feature_engineering import FeatureEngineer
from src.data_fetcher import OpenMeteoFetcher
from datetime import datetime, timedelta


def load_validation_data(days: int, local_file: str = None) -> pd.DataFrame:
    """
    Load validation data from local CSV first, then fall back to API.

    Args:
        days: Number of recent days to use for validation.
        local_file: Path to local CSV (e.g. engineered features or raw data).

    Returns:
        DataFrame with validation data (raw, not yet feature-engineered).
    """
    # --- Strategy 1: use local data files that were already fetched ---
    local_paths = [
        Path(local_file) if local_file else None,
        Path("data/processed/engineered_features.csv"),
        Path("data/raw/historical_data.csv"),
    ]

    for path in local_paths:
        if path and path.exists():
            logger.info(f"Loading validation data from local file: {path}")
            df = pd.read_csv(path)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                # Filter to most recent N days
                cutoff = df['time'].max() - timedelta(days=days)
                df = df[df['time'] >= cutoff].copy()
                if not df.empty:
                    logger.info(f"Using {len(df)} records from {path} (last {days} days)")
                    return df
                else:
                    logger.warning(f"No records in last {days} days in {path}")
            else:
                logger.warning(f"No 'time' column in {path}")

    # --- Strategy 2: fetch from API (fallback) ---
    logger.info(f"No suitable local data found. Fetching {days} days from API...")
    fetcher = OpenMeteoFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = fetcher.fetch_air_quality_history(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    return df


def main():
    """Validate trained models with recent data"""
    parser = argparse.ArgumentParser(description='Validate trained AQI prediction models')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of recent data to validate on'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Optional local CSV file to use for validation data'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model validation")
    
    # Load models
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.info("Please train models first: python scripts/run_training_with_hopsworks.py")
        sys.exit(1)
    
    trainer = ModelTrainer()
    results = trainer.load_models(model_dir)
    
    if not results['models']:
        logger.error("No models found in directory")
        sys.exit(1)
    
    logger.info(f"Loaded {len(results['models'])} models")
    
    # Fetch validation data (local first, then API)
    logger.info(f"Loading {args.days} days of recent data for validation")
    df = load_validation_data(args.days, args.input)
    
    if df is None or df.empty:
        logger.error("No validation data available")
        sys.exit(1)
    
    logger.info(f"Validation data: {len(df)} records")
    
    # Engineer features (only if raw data doesn't already have engineered columns)
    engineer = FeatureEngineer()
    
    # Check if data already has engineered features (e.g. lag columns)
    lag_cols = [c for c in df.columns if '_lag_' in c or '_rolling_' in c]
    if lag_cols:
        logger.info("Data already contains engineered features, skipping re-engineering")
        df_engineered = df
    else:
        df_engineered = engineer.engineer_features(df)
    
    # Prepare data
    X, y, feature_names = engineer.prepare_training_data(df_engineered, target_col='aqi')
    
    if y is None:
        logger.error("No target values in validation data")
        sys.exit(1)
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Valid samples for validation: {len(y)}")
    
    if len(y) == 0:
        logger.error("No valid samples after filtering NaN targets")
        sys.exit(1)
    
    # Validate each model
    logger.info("\n=== Model Validation Results ===")
    
    all_pass = True
    for model_name in results['models'].keys():
        logger.info(f"\nValidating {model_name}...")
        
        try:
            predictions = trainer.predict(X, model_name=model_name)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - y))
            rmse = np.sqrt(np.mean((predictions - y) ** 2))
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2))
            
            logger.info(f"  MAE: {mae:.2f}")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  R²: {r2:.4f}")
            
            # Check if model meets success criteria
            if mae < 15 and r2 > 0.6:
                logger.info(f"  ✓ {model_name} meets success criteria (MAE < 15, R² > 0.6)")
            else:
                logger.warning(f"  ✗ {model_name} does not meet success criteria")
                all_pass = False
            
        except Exception as e:
            logger.error(f"  Error validating {model_name}: {e}")
            all_pass = False
    
    logger.info("\n=== Validation Complete ===")
    
    if not all_pass:
        logger.warning("Some models did not meet success criteria, but validation completed")


if __name__ == "__main__":
    main()
