"""Train models with Hopsworks feature store integration
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.feature_store import FeatureStore

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Feature explanations disabled.")


def main():
    """Train AQI prediction models"""
    parser = argparse.ArgumentParser(description='Train AQI prediction models')
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/engineered_features.csv',
        help='Input CSV file with engineered features'
    )
    parser.add_argument(
        '--use-hopsworks',
        action='store_true',
        help='Load features from Hopsworks feature store'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='Output directory for trained models'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model training pipeline")
    
    # Load data
    df = pd.DataFrame()

    if args.use_hopsworks:
        logger.info("Loading features from Hopsworks feature store")
        try:
            feature_store = FeatureStore()
            df = feature_store.load_features()
            if not df.empty:
                logger.info(f"Loaded {len(df)} records from Hopsworks")
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                # Validate that data is actually numeric (not bracket-notation strings)
                non_time_cols = [c for c in df.columns if c != 'time']
                n_str_cols = sum(1 for c in non_time_cols if df[c].dtype == object)
                if n_str_cols > 0:
                    logger.warning(
                        f"Hopsworks returned {n_str_cols} string columns – "
                        "data may be corrupted, falling back to local CSV"
                    )
                    df = pd.DataFrame()  # Force fallback
            else:
                logger.warning("Hopsworks returned empty DataFrame")
        except Exception as e:
            logger.warning(f"Failed to load from Hopsworks: {e}")

    # Fall back to local CSV if Hopsworks data is empty or not requested
    if df.empty:
        input_path = Path(args.input)
        if args.use_hopsworks:
            logger.info(f"Falling back to local CSV: {input_path}")
        else:
            logger.info(f"Loading features from {input_path}")

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            logger.info("Please run feature pipeline first: python scripts/run_feature_pipeline.py")
            sys.exit(1)

        df = pd.read_csv(input_path)
        df['time'] = pd.to_datetime(df['time'])

        # Sanitize bracket-wrapped scientific notation strings in all columns except 'time'
        import re
        bracket_pattern = re.compile(r'^\[(.+)\]$')
        for col in df.columns:
            if col == 'time':
                continue
            def _parse_value(v):
                if isinstance(v, str):
                    m = bracket_pattern.match(v.strip())
                    if m:
                        try:
                            return float(m.group(1))
                        except ValueError:
                            return float('nan')
                    try:
                        return float(v)
                    except ValueError:
                        return float('nan')
                return v
            df[col] = df[col].apply(_parse_value)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Prepare training data
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.prepare_training_data(df, target_col='aqi')
    
    if y is None:
        logger.error("Target column 'aqi' not found in data")
        sys.exit(1)
    
    logger.info(f"Training data: {X.shape}, Target: {y.shape}")
    logger.info(f"Features: {len(feature_names)}")
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X, y)

    # Sanitize feature matrix X for SHAP (bracket-wrapped scientific notation)
    import re
    bracket_pattern = re.compile(r'^\[(.+)\]$')
    for col in X.columns:
        def _parse_value(v):
            if isinstance(v, str):
                m = bracket_pattern.match(v.strip())
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        return float('nan')
                try:
                    return float(v)
                except ValueError:
                    return float('nan')
            return v
        X[col] = X[col].apply(_parse_value)
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Debug print: show first row and column types before SHAP
    logger.info("\n--- DEBUG: X sample before SHAP ---")
    logger.info(f"First row: {X.iloc[0].to_dict()}")
    logger.info(f"Column types: {X.dtypes}")

    # Print unique values in each column of X_sample
    logger.info("\n--- DEBUG: Unique values in X_sample before SHAP ---")
    X_sample = X.sample(n=min(500, len(X)), random_state=42)
    for col in X_sample.columns:
        uniques = X_sample[col].unique()
        if any(isinstance(u, str) and u.startswith('[') for u in uniques):
            logger.warning(f"Column '{col}' contains bracket-wrapped strings: {uniques}")
        elif any(isinstance(u, str) for u in uniques):
            logger.warning(f"Column '{col}' contains string values: {uniques}")
        else:
            logger.info(f"Column '{col}' unique values: {uniques[:5]} (total: {len(uniques)})")
    
    # Display results
    logger.info("\n=== Model Performance Summary ===")
    for model_name, metrics in results['metrics'].items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
    
    # Find best model
    best_model = min(results['metrics'].items(), key=lambda x: x[1]['mae'])
    logger.info(f"\nBest Model (by MAE): {best_model[0]}")
    logger.info(f"  MAE: {best_model[1]['mae']:.2f}")
    logger.info(f"  R²: {best_model[1]['r2']:.4f}")
    
    # Save models locally
    output_dir = Path(args.output_dir)
    trainer.save_models(results, output_dir)
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Models saved to: {output_dir}")
    
    # Upload models to Hopsworks Model Registry
    if args.use_hopsworks:
        try:
            logger.info("Uploading models to Hopsworks Model Registry...")
            feature_store = FeatureStore()
            
            # Prepare best model metrics for registry
            best_model_name = min(results['metrics'].items(), key=lambda x: x[1]['mae'])[0]
            best_metrics = results['metrics'][best_model_name]
            registry_metrics = {
                "mae": float(best_metrics['mae']),
                "rmse": float(best_metrics['rmse']),
                "r2": float(best_metrics['r2']),
            }
            
            success = feature_store.upload_model_to_registry(
                model_dir=str(output_dir),
                model_name="aqi_prediction_model",
                metrics=registry_metrics,
                description=f"AQI prediction models (best: {best_model_name}, MAE: {best_metrics['mae']:.2f})"
            )
            if success:
                logger.info("Models uploaded to Hopsworks Model Registry")
            else:
                logger.warning("Failed to upload models to Hopsworks (local models still available)")
        except Exception as e:
            logger.warning(f"Hopsworks model registry upload failed: {e}")
    
    # Show feature importance for Random Forest
    if 'random_forest' in results['models']:
        logger.info("\n=== Top 10 Feature Importances (Random Forest) ===")
        importance_df = trainer.get_feature_importance('random_forest')
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # SHAP Analysis for model explainability


if __name__ == "__main__":
    main()
