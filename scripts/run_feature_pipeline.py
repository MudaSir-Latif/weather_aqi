"""
Run feature engineering pipeline
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import OpenMeteoFetcher
from src.feature_engineering import FeatureEngineer
from src.feature_store import FeatureStore


def main():
    """Run feature engineering pipeline"""
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input CSV file (if not provided, fetches fresh data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/engineered_features.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days of data to fetch (if no input file)'
    )
    parser.add_argument(
        '--use-hopsworks',
        action='store_true',
        help='Save features to Hopsworks feature store'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting feature engineering pipeline")
    
    # Load or fetch data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        df['time'] = pd.to_datetime(df['time'])
    else:
        logger.info(f"Fetching {args.days} days of fresh data")
        fetcher = OpenMeteoFetcher()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        df = fetcher.fetch_air_quality_history(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if df.empty:
            logger.error("No data fetched. Pipeline failed.")
            sys.exit(1)
    
    logger.info(f"Input data shape: {df.shape}")
    
    # Engineer features
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    logger.info(f"Engineered features shape: {df_engineered.shape}")
    logger.info(f"Number of features: {len(df_engineered.columns)}")
    
    # Save to Hopsworks if requested; otherwise save locally
    if args.use_hopsworks:
        logger.info("Saving features to Hopsworks (skipping local CSV)")
        feature_store = FeatureStore()
        success = feature_store.save_features(df_engineered)

        if success:
            logger.info("Features saved to Hopsworks successfully")
        else:
            logger.warning("Failed to save features to Hopsworks")
            # Fallback: write local CSV so pipeline output is available
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_engineered.to_csv(output_path, index=False)
            logger.info(f"Saved engineered features to {output_path} (fallback)")
    else:
        # Save to local file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_engineered.to_csv(output_path, index=False)
        logger.info(f"Saved engineered features to {output_path}")
    
    logger.info("=== Feature Engineering Pipeline Complete ===")


if __name__ == "__main__":
    main()
