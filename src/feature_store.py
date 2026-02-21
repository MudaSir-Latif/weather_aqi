"""Feature store integration with Hopsworks
Manages feature groups for AQI prediction in the Hopsworks feature store.
"""
import pandas as pd
from typing import Optional
from loguru import logger
from src.config import HopsworksConfig

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    logger.warning("Hopsworks not installed. Feature store functionality disabled.")


class FeatureStore:
    """Manage features in Hopsworks Feature Store"""
    
    def __init__(self):
        self.config = HopsworksConfig()
        self.project = None
        self.fs = None
        self.feature_group = None
        
        if not HOPSWORKS_AVAILABLE:
            logger.warning("Hopsworks not available")
            return
        
        if self.config.api_key:
            self.connect()
        else:
            logger.info("Hopsworks API key not configured. Skipping connection.")
    
    def connect(self) -> bool:
        """
        Connect to Hopsworks
        
        Returns:
            True if connection successful
        """
        if not HOPSWORKS_AVAILABLE:
            logger.warning("Hopsworks not available")
            return False
        
        try:
            logger.info(f"Connecting to Hopsworks project: {self.config.project_name}")
            
            self.project = hopsworks.login(
                api_key_value=self.config.api_key,
                project=self.config.project_name
            )
            
            self.fs = self.project.get_feature_store()
            
            logger.info("Connected to Hopsworks successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            return False
    
    def _prepare_df_for_hopsworks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for Hopsworks ingestion.
        - Ensures 'time' column is proper datetime
        - Drops non-numeric categorical columns that Hopsworks cannot handle
        - Sanitises column names (lowercase, no special chars)
        """
        df = df.copy()

        # Ensure time is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        # Drop string/object columns that are not the primary key
        # Hopsworks feature groups work best with numeric + datetime columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        drop_cols = [c for c in cat_cols if c not in ('time',)]
        if drop_cols:
            logger.info(f"Dropping non-numeric columns for Hopsworks: {drop_cols}")
            df = df.drop(columns=drop_cols)

        # Sanitise column names: lowercase, replace dots/spaces with underscores
        df.columns = [
            c.lower().replace('.', '_').replace(' ', '_') for c in df.columns
        ]

        return df

    def create_feature_group(
        self, 
        df: pd.DataFrame,
        primary_key: list = None,
        event_time: str = "time"
    ) -> bool:
        """
        Create or update feature group
        
        Args:
            df: DataFrame with features
            primary_key: Primary key columns
            event_time: Event time column
        
        Returns:
            True if successful
        """
        if not self.fs:
            logger.warning("Not connected to Hopsworks")
            return False
        
        try:
            if primary_key is None:
                primary_key = ["time"]

            # Prepare data for Hopsworks
            df = self._prepare_df_for_hopsworks(df)
            
            logger.info(f"Creating/updating feature group: {self.config.feature_group_name}")
            logger.info(f"DataFrame shape for Hopsworks: {df.shape}")
            
            # Get or create feature group
            feature_group = None
            try:
                feature_group = self.fs.get_feature_group(
                    name=self.config.feature_group_name,
                    version=self.config.feature_group_version
                )
            except Exception:
                pass

            if feature_group is not None:
                logger.info("Feature group exists, will update")
            else:
                logger.info("Feature group does not exist, creating new one")
                feature_group = self.fs.create_feature_group(
                    name=self.config.feature_group_name,
                    version=self.config.feature_group_version,
                    primary_key=primary_key,
                    event_time=event_time,
                    description="AQI prediction features for Karachi air quality"
                )
                logger.info("Created new feature group")
            
            if feature_group is None:
                logger.error("Feature group creation returned None - cannot insert data")
                return False
            
            self.feature_group = feature_group
            
            # Insert data
            feature_group.insert(df, write_options={"wait_for_job": True})
            
            logger.info(f"Inserted {len(df)} records into feature group")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/update feature group: {e}")
            return False
    
    def read_feature_group(
        self, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read data from feature group
        
        Args:
            start_time: Start time filter
            end_time: End time filter
        
        Returns:
            DataFrame with features
        """
        if not self.fs:
            logger.warning("Not connected to Hopsworks")
            return pd.DataFrame()
        
        try:
            logger.info(f"Reading feature group: {self.config.feature_group_name}")
            
            feature_group = self.fs.get_feature_group(
                name=self.config.feature_group_name,
                version=self.config.feature_group_version
            )
            
            if feature_group is None:
                logger.warning("Feature group not found")
                return pd.DataFrame()
            
            # Read data directly from feature group
            # (avoids Arrow Flight query.select_all().read() encode bug)
            try:
                df = feature_group.read(read_options={"use_hive": True})
            except Exception as read_err:
                logger.warning(f"Hive read failed ({read_err}), trying default read")
                try:
                    df = feature_group.read()
                except Exception as read_err2:
                    logger.warning(f"Default read also failed ({read_err2}), trying select_all")
                    query = feature_group.select_all()
                    df = query.read(read_options={"use_hive": True})
            
            if start_time and end_time:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    mask = (df['time'] >= start_time) & (df['time'] <= end_time)
                    df = df[mask]
            
            logger.info(f"Read {len(df)} records from feature group")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read feature group: {e}")
            return pd.DataFrame()
    
    def save_features(self, df: pd.DataFrame) -> bool:
        """
        Save features to feature store
        
        Args:
            df: DataFrame with features
        
        Returns:
            True if successful
        """
        return self.create_feature_group(df)
    
    def load_features(
        self, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load features from feature store
        
        Args:
            start_time: Start time filter
            end_time: End time filter
        
        Returns:
            DataFrame with features
        """
        return self.read_feature_group(start_time, end_time)

    def upload_model_to_registry(
        self,
        model_dir: str,
        model_name: str = "aqi_prediction_model",
        model_version: int = None,
        metrics: dict = None,
        description: str = "AQI prediction model for Karachi"
    ) -> bool:
        """
        Upload trained model to Hopsworks Model Registry
        
        Args:
            model_dir: Local directory containing model files
            model_name: Name for the model in the registry
            model_version: Version number (auto-incremented if None)
            metrics: Dictionary of model metrics
            description: Model description
        
        Returns:
            True if successful
        """
        if not self.project:
            logger.warning("Not connected to Hopsworks")
            return False
        
        try:
            mr = self.project.get_model_registry()
            
            logger.info(f"Uploading model '{model_name}' to Hopsworks Model Registry")
            
            # Create model entry
            model_kwargs = {
                "name": model_name,
                "description": description,
                "input_example": None,
            }
            if metrics:
                model_kwargs["metrics"] = metrics
            if model_version:
                model_kwargs["version"] = model_version
            
            model = mr.python.create_model(**model_kwargs)
            model.save(model_dir)
            
            logger.info(f"Model '{model_name}' uploaded to Hopsworks Model Registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload model to registry: {e}")
            return False
