"""
Data preprocessing module.
Handles scaling, normalization, and data cleaning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple
from src.utils.config import Logger


class Preprocessor:
    """Data preprocessing and scaling utilities."""
    
    def __init__(self, scaling_method: str = "standardscaler"):
        """
        Initialize Preprocessor.
        
        Args:
            scaling_method: "standardscaler", "minmaxscaler", or "robustscaler"
        """
        self.scaling_method = scaling_method
        self.logger = Logger()
        self.scaler = self._initialize_scaler()
    
    def _initialize_scaler(self):
        """Initialize scaler based on method."""
        if self.scaling_method == "standardscaler":
            return StandardScaler()
        elif self.scaling_method == "minmaxscaler":
            return MinMaxScaler()
        elif self.scaling_method == "robustscaler":
            return RobustScaler()
        else:
            raise ValueError("Unknown scaling method")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform data."""
        self.logger.info(f"Applying {self.scaling_method}")
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, method: str = "iqr", 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Args:
            df: Input DataFrame
            method: "iqr" (Interquartile Range) or "zscore"
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame without outliers
        """
        if method == "iqr":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)
        elif method == "zscore":
            from scipy.stats import zscore
            mask = (np.abs(zscore(df.select_dtypes(include=[np.number]))) < threshold).all(axis=1)
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        removed = len(df) - len(df[mask])
        Logger.info(f"Removed {removed} outliers using {method}")
        return df[mask]
