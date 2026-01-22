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
    
    # Fetch recent data
    logger.info(f"Fetching {args.days} days of recent data for validation")
    fetcher = OpenMeteoFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    df = fetcher.fetch_air_quality_history(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    if df.empty:
        logger.error("No validation data fetched")
        sys.exit(1)
    
    logger.info(f"Validation data: {len(df)} records")
    
    # Engineer features
    engineer = FeatureEngineer()
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
    
    # Validate each model
    logger.info("\n=== Model Validation Results ===")
    
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
            
        except Exception as e:
            logger.error(f"  Error validating {model_name}: {e}")
    
    logger.info("\n=== Validation Complete ===")


if __name__ == "__main__":
    main()
