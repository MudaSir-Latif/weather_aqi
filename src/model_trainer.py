"""
Model training for AQI prediction
Supports Ridge, Random Forest, XGBoost, and LSTM models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from loguru import logger

from src.config import ModelConfig

# Optional deep-learning imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed — LSTM model will be unavailable.")


# ---------------------------------------------------------------------------
# LSTM helpers (defined at module level so they are pickle-friendly)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        """Simple LSTM regressor."""

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])  # last timestep
            return out


class LSTMWrapper:
    """Sklearn-compatible wrapper around a PyTorch LSTM model."""

    def __init__(self, net, input_size: int):
        self.net = net
        self.input_size = input_size

    def predict(self, X) -> np.ndarray:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available.")
        self.net.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            preds = self.net(X_t).squeeze(-1).numpy()
        return preds

    # Needed so joblib can serialize the wrapper
    def __getstate__(self):
        import io
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return {"state_dict": buf.getvalue(), "input_size": self.input_size,
                "hidden_size": self.net.lstm.hidden_size, "num_layers": self.net.lstm.num_layers}

    def __setstate__(self, state):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required to load LSTM model.")
        net = _LSTMNet(state["input_size"], state["hidden_size"], state["num_layers"])
        import io
        buf = io.BytesIO(state["state_dict"])
        net.load_state_dict(torch.load(buf, weights_only=True))
        net.eval()
        self.net = net
        self.input_size = state["input_size"]


class ModelTrainer:
    """Train and evaluate AQI prediction models"""
    
    def __init__(self):
        self.config = ModelConfig()
        self.models = {}
        self.feature_names = []
        self.metrics = {}
        logger.info("Initialized ModelTrainer")
    
    def train_ridge(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        alpha: float = 1.0
    ) -> Ridge:
        """
        Train Ridge Regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
        
        Returns:
            Trained Ridge model
        """
        logger.info(f"Training Ridge Regression (alpha={alpha})")
        
        model = Ridge(alpha=alpha, random_state=self.config.random_state)
        model.fit(X_train, y_train)
        
        logger.info("Ridge Regression training complete")
        return model
    
    def train_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 15
    ) -> RandomForestRegressor:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        
        Returns:
            Trained Random Forest model
        """
        logger.info(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        logger.info("Random Forest training complete")
        return model
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        
        Returns:
            Trained XGBoost model
        """
        logger.info(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth})")
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        logger.info("XGBoost training complete")
        return model
    
    # ------------------------------------------------------------------
    # LSTM Deep Learning model
    # ------------------------------------------------------------------

    def train_lstm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int = None,
        batch_size: int = None,
        hidden_size: int = None,
    ) -> Any:
        """
        Train an LSTM model for AQI prediction.

        The tabular features are reshaped into a single-timestep sequence
        so the LSTM learns a non-linear mapping.  For production you would
        feed rolling windows; this keeps the API identical to the other
        trainers so it plugs into the existing pipeline.

        Returns:
            LSTMWrapper that exposes a sklearn-compatible .predict() method.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed — skipping LSTM training.")
            return None

        epochs = epochs or self.config.lstm_epochs
        batch_size = batch_size or self.config.lstm_batch_size
        hidden_size = hidden_size or self.config.lstm_hidden_size

        logger.info(
            f"Training LSTM (epochs={epochs}, batch={batch_size}, hidden={hidden_size})"
        )

        input_size = X_train.shape[1]

        # Convert to tensors
        X_t = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)  # (N,1,F)
        y_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # (N,1)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build model
        net = _LSTMNet(input_size, hidden_size)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        net.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                logger.info(f"  LSTM epoch {epoch}/{epochs}  loss={epoch_loss / len(dataset):.4f}")

        wrapper = LSTMWrapper(net, input_size)
        logger.info("LSTM training complete")
        return wrapper
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_test_samples': len(y_test)
        }
        
        logger.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return metrics
    
    def train_all_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all configured models
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Dictionary with trained models and metrics
        """
        logger.info("Starting training for all models")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        results = {
            'models': {},
            'metrics': {},
            'feature_names': self.feature_names
        }
        
        # Train Ridge Regression
        if 'ridge' in self.config.models_to_train:
            ridge_model = self.train_ridge(X_train, y_train)
            ridge_metrics = self.evaluate_model(ridge_model, X_test, y_test, 'Ridge')
            results['models']['ridge'] = ridge_model
            results['metrics']['ridge'] = ridge_metrics
        
        # Train Random Forest
        if 'random_forest' in self.config.models_to_train:
            rf_model = self.train_random_forest(X_train, y_train)
            rf_metrics = self.evaluate_model(rf_model, X_test, y_test, 'RandomForest')
            results['models']['random_forest'] = rf_model
            results['metrics']['random_forest'] = rf_metrics
        
        # Train XGBoost
        if 'xgboost' in self.config.models_to_train:
            xgb_model = self.train_xgboost(X_train, y_train)
            xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
            results['models']['xgboost'] = xgb_model
            results['metrics']['xgboost'] = xgb_metrics
        
        # Train LSTM (deep learning)
        if 'lstm' in self.config.models_to_train:
            if TORCH_AVAILABLE:
                lstm_model = self.train_lstm(X_train, y_train)
                if lstm_model is not None:
                    lstm_metrics = self.evaluate_model(lstm_model, X_test, y_test, 'LSTM')
                    results['models']['lstm'] = lstm_model
                    results['metrics']['lstm'] = lstm_metrics
            else:
                logger.warning("Skipping LSTM — PyTorch not installed.")
        
        # Store in instance
        self.models = results['models']
        self.metrics = results['metrics']
        
        logger.info("All models trained successfully")
        return results
    
    def save_models(self, results: Dict[str, Any], output_dir: Path = None) -> None:
        """
        Save trained models to disk
        
        Args:
            results: Dictionary with models and metadata
            output_dir: Output directory (default from config)
        """
        if output_dir is None:
            output_dir = self.config.model_path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {output_dir}")
        
        # Save each model
        for model_name, model in results['models'].items():
            model_file = output_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save feature names
        feature_file = output_dir / "feature_names.joblib"
        joblib.dump(results['feature_names'], feature_file)
        logger.info(f"Saved feature names to {feature_file}")
        
        # Save metrics
        metrics_file = output_dir / "model_metrics.joblib"
        joblib.dump(results['metrics'], metrics_file)
        logger.info(f"Saved metrics to {metrics_file}")
    
    def load_models(self, model_dir: Path = None) -> Dict[str, Any]:
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
        
        Returns:
            Dictionary with models and metadata
        """
        if model_dir is None:
            model_dir = self.config.model_path
        
        model_dir = Path(model_dir)
        
        logger.info(f"Loading models from {model_dir}")
        
        results = {
            'models': {},
            'metrics': {},
            'feature_names': []
        }
        
        # Load each model
        for model_name in self.config.models_to_train:
            model_file = model_dir / f"{model_name}_model.joblib"
            if model_file.exists():
                results['models'][model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name}")
        
        # Load feature names
        feature_file = model_dir / "feature_names.joblib"
        if feature_file.exists():
            results['feature_names'] = joblib.load(feature_file)
            self.feature_names = results['feature_names']
            logger.info("Loaded feature names")
        
        # Load metrics
        metrics_file = model_dir / "model_metrics.joblib"
        if metrics_file.exists():
            results['metrics'] = joblib.load(metrics_file)
            self.metrics = results['metrics']
            logger.info("Loaded metrics")
        
        self.models = results['models']
        
        return results
    
    def predict(
        self, 
        X: pd.DataFrame, 
        model_name: str = 'ridge'
    ) -> np.ndarray:
        """
        Make predictions using specified model
        
        Args:
            X: Features
            model_name: Model to use for prediction
        
        Returns:
            Array of predictions
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found. Available: {list(self.models.keys())}")
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        return predictions
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            model_name: Model name (random_forest or xgboost)
        
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature importances")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
