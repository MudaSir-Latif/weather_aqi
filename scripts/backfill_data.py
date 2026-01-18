"""
Backfill historical air quality data
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import OpenMeteoFetcher
from src.config import LocationConfig


def main():
    """Backfill historical data"""
    parser = argparse.ArgumentParser(description='Backfill historical air quality data')
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days to backfill (default: 365)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/historical_data.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting backfill for {args.days} days")
    logger.info(f"Output file: {args.output}")
    
    # Initialize fetcher
    fetcher = OpenMeteoFetcher()
    location = LocationConfig()
    
    logger.info(f"Location: {location.city_name} ({location.latitude}, {location.longitude})")
    
    # Fetch historical data
    df = fetcher.backfill_historical_data(days=args.days)
    
    if df.empty:
        logger.error("No data fetched. Backfill failed.")
        sys.exit(1)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")
    
    # Print summary statistics
    logger.info("\n=== Data Summary ===")
    logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    if 'aqi' in df.columns:
        logger.info(f"\nAQI Statistics:")
        logger.info(f"  Mean: {df['aqi'].mean():.2f}")
        logger.info(f"  Median: {df['aqi'].median():.2f}")
        logger.info(f"  Min: {df['aqi'].min():.2f}")
        logger.info(f"  Max: {df['aqi'].max():.2f}")
        logger.info(f"  Std: {df['aqi'].std():.2f}")
        
        # Category distribution
        if 'aqi_category' in df.columns:
            logger.info(f"\nAQI Category Distribution:")
            category_counts = df['aqi_category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    logger.info("\n=== Backfill Complete ===")


if __name__ == "__main__":
    main()
