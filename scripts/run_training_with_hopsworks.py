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
    if SHAP_AVAILABLE:
        logger.info("\n=== SHAP Feature Explanations ===")
        try:
            # Use a sample of test data for SHAP (faster computation)
            X_sample = X.sample(n=min(500, len(X)), random_state=42)
            
            # SHAP for XGBoost (best model)
            if 'xgboost' in results['models']:
                logger.info("Computing SHAP values for XGBoost model...")
                xgb_model = results['models']['xgboost']
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(X_sample)
                
                # Get mean absolute SHAP values per feature
                shap_importance = pd.DataFrame({
                    'feature': feature_names,
                    'mean_shap': np.abs(shap_values).mean(axis=0)
                }).sort_values('mean_shap', ascending=False)
                
                logger.info("\nTop 15 SHAP Feature Importances (XGBoost):")
                for idx, row in shap_importance.head(15).iterrows():
                    logger.info(f"  {row['feature']}: {row['mean_shap']:.4f}")
                
                # Save SHAP summary plot
                output_dir_path = Path(args.output_dir)
                shap_dir = output_dir_path / "shap"
                shap_dir.mkdir(parents=True, exist_ok=True)
                
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # Summary bar plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                  plot_type="bar", show=False, max_display=20)
                plt.tight_layout()
                plt.savefig(shap_dir / "shap_summary_bar.png", dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved SHAP bar plot to {shap_dir / 'shap_summary_bar.png'}")
                
                # Summary dot plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                  show=False, max_display=20)
                plt.tight_layout()
                plt.savefig(shap_dir / "shap_summary_dot.png", dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved SHAP dot plot to {shap_dir / 'shap_summary_dot.png'}")
                
                # Save SHAP values for later use (e.g., dashboard)
                shap_importance.to_csv(shap_dir / "shap_feature_importance.csv", index=False)
                logger.info(f"Saved SHAP importance to {shap_dir / 'shap_feature_importance.csv'}")
            
            # SHAP for Random Forest 
            if 'random_forest' in results['models']:
                logger.info("\nComputing SHAP values for Random Forest model...")
                rf_model = results['models']['random_forest']
                rf_explainer = shap.TreeExplainer(rf_model)
                rf_shap_values = rf_explainer.shap_values(X_sample)
                
                rf_shap_importance = pd.DataFrame({
                    'feature': feature_names,
                    'mean_shap': np.abs(rf_shap_values).mean(axis=0)
                }).sort_values('mean_shap', ascending=False)
                
                logger.info("\nTop 15 SHAP Feature Importances (Random Forest):")
                for idx, row in rf_shap_importance.head(15).iterrows():
                    logger.info(f"  {row['feature']}: {row['mean_shap']:.4f}")
                
                rf_shap_importance.to_csv(shap_dir / "shap_rf_feature_importance.csv", index=False)
                
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
    else:
        logger.info("SHAP not available. Install with: pip install shap")


if __name__ == "__main__":
    main()
