"""
Data loading and preprocessing module.
Supports Pandas (small data) and PySpark (large data) backends.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from src.utils.config import Logger


class DataLoader:
    """Load and preprocess data using Pandas or Spark backend."""
    
    def __init__(self, backend: str = "pandas"):
        """
        Initialize DataLoader.
        
        Args:
            backend: "pandas" for small datasets (<5GB), "spark" for large datasets
        """
        if backend not in ["pandas", "spark"]:
            raise ValueError("Backend must be 'pandas' or 'spark'")
        
        self.backend = backend
        self.logger = Logger()
        
        if backend == "spark":
            self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session for distributed processing."""
        try:
            from pyspark.sql import SparkSession
            self.spark = SparkSession.builder \
                .appName("FactoryGuardAI") \
                .config("spark.sql.shuffle.partitions", "200") \
                .getOrCreate()
            self.logger.info("Spark session initialized")
        except ImportError:
            self.logger.error("PySpark not installed. Install with: pip install pyspark")
            raise
    
    def load_csv(self, filepath: str) -> Union[pd.DataFrame, 'pyspark.sql.DataFrame']:
        """
        Load CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame (Pandas or Spark)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.logger.info(f"Loading data from {filepath}")
        
        if self.backend == "pandas":
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        else:
            df = self.spark.read.csv(str(filepath), header=True, inferSchema=True)
            row_count = df.count()
            col_count = len(df.columns)
            self.logger.info(f"Loaded {row_count} rows, {col_count} columns")
            return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: "mean", "median", or "drop"
            
        Returns:
            DataFrame with missing values handled
        """
        self.logger.info(f"Handling missing values with strategy: {strategy}")
        
        if strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "drop":
            df = df.dropna()
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_len = len(df)
        df = df.drop_duplicates()
        removed = original_len - len(df)
        self.logger.info(f"Removed {removed} duplicate rows")
        return df
    
    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test set (0-1)
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split as sklearn_split
        
        train_df, test_df = sklearn_split(df, test_size=test_size, 
                                         random_state=random_state)
        
        self.logger.info(f"Split data: Train={len(train_df)}, Test={len(test_df)}")
        return train_df, test_df
