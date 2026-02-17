"""
Data fetcher for Open-Meteo API
Fetches air quality and weather data without requiring API keys
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time

from src.config import OpenMeteoConfig, LocationConfig


class AQICalculator:
    """Calculate Air Quality Index using US EPA standard"""
    
    @staticmethod
    def calculate_pm25_aqi(pm25: float) -> int:
        """Calculate AQI for PM2.5"""
        if pd.isna(pm25) or pm25 < 0:
            return 0
        
        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500)
        ]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= pm25 <= c_high:
                return int(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
        
        return 500  # Hazardous
    
    @staticmethod
    def calculate_pm10_aqi(pm10: float) -> int:
        """Calculate AQI for PM10"""
        if pd.isna(pm10) or pm10 < 0:
            return 0
        
        breakpoints = [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 604, 301, 500)
        ]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= pm10 <= c_high:
                return int(((i_high - i_low) / (c_high - c_low)) * (pm10 - c_low) + i_low)
        
        return 500
    
    @staticmethod
    def calculate_o3_aqi(o3: float) -> int:
        """Calculate AQI for Ozone (8-hour average in µg/m³)"""
        if pd.isna(o3) or o3 < 0:
            return 0
        
        # Convert µg/m³ to ppm (at 25°C): ppm = µg/m³ * 0.0005
        o3_ppm = o3 * 0.0005
        
        breakpoints = [
            (0.000, 0.054, 0, 50),
            (0.055, 0.070, 51, 100),
            (0.071, 0.085, 101, 150),
            (0.086, 0.105, 151, 200),
            (0.106, 0.200, 201, 300)
        ]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= o3_ppm <= c_high:
                return int(((i_high - i_low) / (c_high - c_low)) * (o3_ppm - c_low) + i_low)
        
        return 300
    
    @staticmethod
    def calculate_no2_aqi(no2: float) -> int:
        """Calculate AQI for NO2 (ppb)"""
        if pd.isna(no2) or no2 < 0:
            return 0
        
        # Convert µg/m³ to ppb if needed: ppb = µg/m³ * 0.5319
        no2_ppb = no2 * 0.5319 if no2 > 1000 else no2
        
        breakpoints = [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 2049, 301, 500)
        ]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= no2_ppb <= c_high:
                return int(((i_high - i_low) / (c_high - c_low)) * (no2_ppb - c_low) + i_low)
        
        return 500
    
    @staticmethod
    def calculate_so2_aqi(so2: float) -> int:
        """Calculate AQI for SO2 (ppb)"""
        if pd.isna(so2) or so2 < 0:
            return 0
        
        # Convert µg/m³ to ppb if needed: ppb = µg/m³ * 0.382
        so2_ppb = so2 * 0.382 if so2 > 1000 else so2
        
        breakpoints = [
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 1004, 301, 500)
        ]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= so2_ppb <= c_high:
                return int(((i_high - i_low) / (c_high - c_low)) * (so2_ppb - c_low) + i_low)
        
        return 500
    
    @staticmethod
    def calculate_co_aqi(co: float) -> int:
        """Calculate AQI for CO (ppm)"""
        if pd.isna(co) or co < 0:
            return 0
        
        # Convert µg/m³ to ppm if needed: ppm = µg/m³ * 0.000873
        co_ppm = co * 0.000873 if co > 100 else co
        
        breakpoints = [
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 50.4, 301, 500)
        ]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= co_ppm <= c_high:
                return int(((i_high - i_low) / (c_high - c_low)) * (co_ppm - c_low) + i_low)
        
        return 500
    
    @classmethod
    def calculate_aqi(cls, pollutants: Dict[str, float]) -> Tuple[int, str]:
        """
        Calculate overall AQI from all pollutants
        Returns: (aqi_value, dominant_pollutant)
        """
        aqi_values = {}
        
        if 'pm2_5' in pollutants and pollutants['pm2_5'] is not None:
            aqi_values['PM2.5'] = cls.calculate_pm25_aqi(pollutants['pm2_5'])
        
        if 'pm10' in pollutants and pollutants['pm10'] is not None:
            aqi_values['PM10'] = cls.calculate_pm10_aqi(pollutants['pm10'])
        
        if 'o3' in pollutants and pollutants['o3'] is not None:
            aqi_values['O3'] = cls.calculate_o3_aqi(pollutants['o3'])
        
        if 'no2' in pollutants and pollutants['no2'] is not None:
            aqi_values['NO2'] = cls.calculate_no2_aqi(pollutants['no2'])
        
        if 'so2' in pollutants and pollutants['so2'] is not None:
            aqi_values['SO2'] = cls.calculate_so2_aqi(pollutants['so2'])
        
        if 'co' in pollutants and pollutants['co'] is not None:
            aqi_values['CO'] = cls.calculate_co_aqi(pollutants['co'])
        
        if not aqi_values:
            return 0, "Unknown"
        
        # Overall AQI is the maximum of all sub-indices
        max_aqi = max(aqi_values.values())
        dominant_pollutant = max(aqi_values, key=aqi_values.get)
        
        return max_aqi, dominant_pollutant
    
    @staticmethod
    def get_aqi_category(aqi: int) -> str:
        """Get AQI category and health message"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"


class OpenMeteoFetcher:
    """Fetch air quality and weather data from Open-Meteo API"""
    
    def __init__(self):
        self.config = OpenMeteoConfig()
        self.location = LocationConfig()
        self.aqi_calc = AQICalculator()
        logger.info(f"Initialized OpenMeteoFetcher for {self.location.city_name}")
    
    def fetch_air_quality_history(
        self, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical air quality data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with air quality data
        """
        try:
            params = {
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "hourly": ",".join(self.config.air_quality_params),
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "auto"
            }
            
            logger.info(f"Fetching air quality history from {start_date} to {end_date}")
            
            # Retry logic for resilience against transient network issues
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(
                        self.config.air_quality_url,
                        params=params,
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout on attempt {attempt}/{max_retries}")
                    if attempt == max_retries:
                        raise
                    time.sleep(5 * attempt)
                except requests.exceptions.ConnectionError:
                    logger.warning(f"Connection error on attempt {attempt}/{max_retries}")
                    if attempt == max_retries:
                        raise
                    time.sleep(5 * attempt)
            
            data = response.json()
            
            if 'hourly' not in data:
                logger.error("No hourly data in response")
                return pd.DataFrame()
            
            # Parse hourly data
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'carbon_monoxide': 'co',
                'nitrogen_dioxide': 'no2',
                'sulphur_dioxide': 'so2',
                'ozone': 'o3',
                'ammonia': 'nh3'
            })
            
            # Calculate AQI for each row
            logger.info("Calculating AQI values")
            aqi_data = []
            for idx, row in df.iterrows():
                pollutants = {
                    'pm2_5': row.get('pm2_5', 0),
                    'pm10': row.get('pm10', 0),
                    'co': row.get('co', 0),
                    'no2': row.get('no2', 0),
                    'so2': row.get('so2', 0),
                    'o3': row.get('o3', 0)
                }
                aqi, dominant = self.aqi_calc.calculate_aqi(pollutants)
                aqi_data.append({
                    'aqi': aqi,
                    'dominant_pollutant': dominant,
                    'aqi_category': self.aqi_calc.get_aqi_category(aqi)
                })
            
            aqi_df = pd.DataFrame(aqi_data)
            df = pd.concat([df, aqi_df], axis=1)
            
            logger.info(f"Fetched {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching air quality history: {e}")
            return pd.DataFrame()
    
    def fetch_current_air_quality(self) -> Dict:
        """
        Fetch current air quality data
        
        Returns:
            Dictionary with current air quality metrics
        """
        try:
            # Fetch current day data
            today = datetime.now().strftime("%Y-%m-%d")
            
            params = {
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "hourly": ",".join(self.config.air_quality_params),
                "start_date": today,
                "end_date": today,
                "timezone": "auto"
            }
            
            logger.info("Fetching current air quality")
            response = requests.get(
                self.config.air_quality_url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'hourly' not in data or not data['hourly']['time']:
                logger.error("No current data available")
                return {}
            
            # Get the most recent hour
            latest_idx = -1
            hourly_data = data['hourly']
            
            current_data = {
                'timestamp': hourly_data['time'][latest_idx],
                'location': self.location.city_name,
                'latitude': self.location.latitude,
                'longitude': self.location.longitude,
                'pm10': hourly_data.get('pm10', [None])[latest_idx],
                'pm2_5': hourly_data.get('pm2_5', [None])[latest_idx],
                'co': hourly_data.get('carbon_monoxide', [None])[latest_idx],
                'no2': hourly_data.get('nitrogen_dioxide', [None])[latest_idx],
                'so2': hourly_data.get('sulphur_dioxide', [None])[latest_idx],
                'o3': hourly_data.get('ozone', [None])[latest_idx],
                'nh3': hourly_data.get('ammonia', [None])[latest_idx],
                'dust': hourly_data.get('dust', [None])[latest_idx],
                'aerosol_optical_depth': hourly_data.get('aerosol_optical_depth', [None])[latest_idx]
            }
            
            # Calculate AQI
            pollutants = {
                'pm2_5': current_data['pm2_5'],
                'pm10': current_data['pm10'],
                'co': current_data['co'],
                'no2': current_data['no2'],
                'so2': current_data['so2'],
                'o3': current_data['o3']
            }
            
            aqi, dominant = self.aqi_calc.calculate_aqi(pollutants)
            current_data['aqi'] = aqi
            current_data['dominant_pollutant'] = dominant
            current_data['aqi_category'] = self.aqi_calc.get_aqi_category(aqi)
            
            logger.info(f"Current AQI: {aqi} ({current_data['aqi_category']})")
            return current_data
            
        except Exception as e:
            logger.error(f"Error fetching current air quality: {e}")
            return {}
    
    def fetch_weather_forecast(self, forecast_days: int = 3) -> pd.DataFrame:
        """
        Fetch weather forecast data
        
        Args:
            forecast_days: Number of days to forecast (default 3)
        
        Returns:
            DataFrame with weather forecast
        """
        try:
            params = {
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "hourly": ",".join(self.config.weather_params),
                "forecast_days": forecast_days,
                "timezone": "auto"
            }
            
            logger.info(f"Fetching {forecast_days}-day weather forecast")
            response = requests.get(
                self.config.weather_url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'hourly' not in data:
                logger.error("No hourly weather data in response")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            
            logger.info(f"Fetched {len(df)} weather forecast records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return pd.DataFrame()
    
    def backfill_historical_data(self, days: int = 365) -> pd.DataFrame:
        """
        Backfill historical air quality and weather data
        
        Args:
            days: Number of days to backfill (default 365)
        
        Returns:
            Combined DataFrame with air quality and weather data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Backfilling {days} days of historical data")
            
            # Fetch in chunks to avoid timeout (90 days at a time)
            chunk_size = 90
            all_data = []
            
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=chunk_size), end_date)
                
                start_str = current_start.strftime("%Y-%m-%d")
                end_str = current_end.strftime("%Y-%m-%d")
                
                logger.info(f"Fetching chunk: {start_str} to {end_str}")
                
                # Fetch air quality
                air_quality_df = self.fetch_air_quality_history(start_str, end_str)
                
                if not air_quality_df.empty:
                    all_data.append(air_quality_df)
                
                # Sleep to avoid rate limiting
                time.sleep(2)
                
                current_start = current_end + timedelta(days=1)
            
            if not all_data:
                logger.error("No data fetched during backfill")
                return pd.DataFrame()
            
            # Combine all chunks
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            logger.info(f"Backfilled {len(combined_df)} total records")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error during backfill: {e}")
            return pd.DataFrame()
    
    def fetch_combined_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch both air quality and weather data for the same period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Combined DataFrame with air quality and weather data
        """
        try:
            # Fetch air quality
            air_quality_df = self.fetch_air_quality_history(start_date, end_date)
            
            # Wait a bit before next request
            time.sleep(1)
            
            # Fetch weather using the forecast endpoint with historical dates
            # Note: Open-Meteo's forecast endpoint only provides future data
            # For historical weather, we would need to use the historical weather API
            # For now, we'll just return air quality data
            
            logger.info(f"Fetched combined data: {len(air_quality_df)} records")
            return air_quality_df
            
        except Exception as e:
            logger.error(f"Error fetching combined data: {e}")
            return pd.DataFrame()
