"""
Feature engineering for AQI prediction
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from loguru import logger
from src.config import ModelConfig


class FeatureEngineer:
    """Feature engineering for AQI prediction models"""
    
    def __init__(self):
        self.config = ModelConfig()
        logger.info("Initialized FeatureEngineer")
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features with cyclical encoding
        
        Args:
            df: DataFrame with 'time' column
        
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        if 'time' not in df.columns:
            logger.warning("No 'time' column found")
            return df
        
        # Ensure datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Extract time components
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        
        # Cyclical encoding for hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Cyclical encoding for day of year (365-day cycle)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        logger.info("Created temporal features")
        return df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        lags: List[int] = None
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods (default from config)
        
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if lags is None:
            lags = [1, 3, 6, 12, 24]  # hourly lags
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"Created lag features for {len(columns)} columns with {len(lags)} lags")
        return df
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Create rolling statistics features
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: Window sizes (default from config)
        
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        if windows is None:
            windows = self.config.rolling_window_sizes
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling min
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).min()
                
                # Rolling max
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).max()
        
        logger.info(f"Created rolling features for {len(columns)} columns with {len(windows)} windows")
        return df
    
    def create_pollutant_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create pollutant interaction features
        
        Args:
            df: DataFrame with pollutant columns
        
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
        
        # Ratios
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
        
        # Combined pollutant index
        pollutant_cols = [col for col in pollutants if col in df.columns]
        if len(pollutant_cols) > 0:
            # Normalize and create combined index
            for col in pollutant_cols:
                if df[col].std() > 0:
                    df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
        
        logger.info("Created pollutant interaction features")
        return df
    
    def create_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather interaction features if weather data is available
        
        Args:
            df: DataFrame with weather columns
        
        Returns:
            DataFrame with weather interaction features
        """
        df = df.copy()
        
        # Temperature and humidity interaction
        if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
            df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
        
        # Wind speed and direction
        if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
            # Convert wind to components
            wind_rad = np.radians(df['wind_direction_10m'])
            df['wind_u'] = df['wind_speed_10m'] * np.sin(wind_rad)
            df['wind_v'] = df['wind_speed_10m'] * np.cos(wind_rad)
        
        logger.info("Created weather interaction features")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline
        
        Args:
            df: Raw DataFrame with air quality data
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        df = df.copy()
        
        # Sort by time
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)
        
        # 1. Temporal features
        df = self.create_temporal_features(df)
        
        # 2. Define pollutant columns
        pollutant_cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        # 3. Lag features
        if available_pollutants:
            df = self.create_lag_features(df, available_pollutants)
        
        # 4. Rolling features
        if available_pollutants:
            df = self.create_rolling_features(df, available_pollutants)
        
        # 5. Pollutant interactions
        df = self.create_pollutant_interactions(df)
        
        # 6. Weather interactions
        df = self.create_weather_interactions(df)
        
        # 7. Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        logger.info(f"Features: {df.columns.tolist()}")
        
        return df
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'aqi'
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with engineered features
            target_col: Target column name
        
        Returns:
            Tuple of (features_df, target_series, feature_names)
        """
        df = df.copy()
        
        # Columns to exclude from features
        exclude_cols = [
            'time', 
            target_col, 
            'aqi_category', 
            'dominant_pollutant',
            'timestamp'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Separate features and target
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        
        logger.info(f"Prepared training data: {len(feature_cols)} features, {len(df)} samples")
        
        return X, y, feature_cols
