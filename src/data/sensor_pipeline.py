"""
Real-Time Sensor Data Pipeline for Manufacturing Plant
Handles live data from 500 robotic arms with vibration, temperature, and pressure sensors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple
import joblib

from src.utils.config import Logger


class SensorDataPipeline:
    """Pipeline for processing real-time sensor data from robotic arms."""
    
    def __init__(self, n_arms: int = 500, scaler_path: str = "models/scaler.pkl"):
        """
        Initialize sensor pipeline.
        
        Args:
            n_arms: Number of robotic arms (default: 500)
            scaler_path: Path to fitted scaler for preprocessing
        """
        self.n_arms = n_arms
        self.logger = Logger()
        self.scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None
        self.arm_ids = [f"ARM_{i:03d}" for i in range(n_arms)]
        self.sensor_history = {}
        
        self.logger.info(f"Initialized sensor pipeline for {n_arms} robotic arms")
    
    def ingest_sensor_data(self, arm_id: str, temperature: float, pressure: float,
                          vibration: float, humidity: float, 
                          power_consumption: float) -> Dict:
        """
        Ingest real-time sensor data from a single robotic arm.
        
        Args:
            arm_id: Robotic arm identifier (e.g., 'ARM_001')
            temperature: Temperature reading (°C)
            pressure: Pressure reading (Pa)
            vibration: Vibration reading (mm/s)
            humidity: Humidity reading (%)
            power_consumption: Power consumption (W)
            
        Returns:
            Dictionary with preprocessed sensor data
        """
        timestamp = datetime.now()
        
        # Create sensor record
        sensor_record = {
            'arm_id': arm_id,
            'timestamp': timestamp,
            'temperature': temperature,
            'pressure': pressure,
            'vibration': vibration,
            'humidity': humidity,
            'power_consumption': power_consumption,
            'raw_received': True
        }
        
        # Store in history for trend analysis
        if arm_id not in self.sensor_history:
            self.sensor_history[arm_id] = []
        
        self.sensor_history[arm_id].append(sensor_record)
        
        # Keep only last 1440 readings (24 hours at 1-minute intervals)
        if len(self.sensor_history[arm_id]) > 1440:
            self.sensor_history[arm_id].pop(0)
        
        return sensor_record
    
    def preprocess_features(self, sensor_data: Dict) -> np.ndarray:
        """
        Preprocess sensor data for model prediction.
        
        Args:
            sensor_data: Raw sensor readings
            
        Returns:
            Scaled feature vector
        """
        features = np.array([
            sensor_data['temperature'],
            sensor_data['pressure'],
            sensor_data['vibration'],
            sensor_data['humidity'],
            sensor_data['power_consumption']
        ]).reshape(1, -1)
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        return features
    
    def calculate_trend_features(self, arm_id: str) -> Dict[str, float]:
        """
        Calculate trend features from historical sensor data.
        
        Args:
            arm_id: Robotic arm identifier
            
        Returns:
            Dictionary with trend metrics
        """
        if arm_id not in self.sensor_history or len(self.sensor_history[arm_id]) < 60:
            return {}
        
        history = self.sensor_history[arm_id]
        df = pd.DataFrame(history[-60:])  # Last 60 readings
        
        # Calculate trend statistics
        trends = {
            'temp_trend': float(df['temperature'].iloc[-1] - df['temperature'].iloc[0]),
            'vibration_trend': float(df['vibration'].iloc[-1] - df['vibration'].iloc[0]),
            'pressure_trend': float(df['pressure'].iloc[-1] - df['pressure'].iloc[0]),
            'temp_std': float(df['temperature'].std()),
            'vibration_std': float(df['vibration'].std()),
            'pressure_std': float(df['pressure'].std()),
            'temp_max': float(df['temperature'].max()),
            'vibration_max': float(df['vibration'].max()),
            'pressure_max': float(df['pressure'].max()),
        }
        
        return trends
    
    def get_health_snapshot(self, arm_id: str) -> Dict:
        """Get current health snapshot for an arm."""
        if arm_id not in self.sensor_history or not self.sensor_history[arm_id]:
            return None
        
        latest = self.sensor_history[arm_id][-1]
        trends = self.calculate_trend_features(arm_id)
        
        return {
            'arm_id': arm_id,
            'timestamp': latest['timestamp'],
            'current_readings': {
                'temperature': latest['temperature'],
                'pressure': latest['pressure'],
                'vibration': latest['vibration'],
                'humidity': latest['humidity'],
                'power_consumption': latest['power_consumption'],
            },
            'trends': trends,
            'data_points_available': len(self.sensor_history[arm_id])
        }
    
    def get_all_health_snapshot(self) -> Dict:
        """Get health snapshot for all arms."""
        snapshots = {}
        for arm_id in self.arm_ids:
            snapshot = self.get_health_snapshot(arm_id)
            if snapshot:
                snapshots[arm_id] = snapshot
        
        return snapshots
    
    def export_batch_data(self, arm_id: str, hours: int = 24) -> pd.DataFrame:
        """
        Export historical sensor data for batch processing.
        
        Args:
            arm_id: Robotic arm identifier
            hours: Number of hours to export
            
        Returns:
            DataFrame with sensor history
        """
        if arm_id not in self.sensor_history:
            return pd.DataFrame()
        
        history = self.sensor_history[arm_id]
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered = [r for r in history if r['timestamp'] >= cutoff_time]
        return pd.DataFrame(filtered)
