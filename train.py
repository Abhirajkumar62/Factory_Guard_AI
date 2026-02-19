"""
Main training script for Factory Guard AI.
Production-ready script for model training and evaluation.
This script demonstrates how to refactor notebook code into production-ready Python.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.data.loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.models.trainer import ModelTrainer
from src.utils.config import ConfigLoader, Logger
from src.utils.mlflow_tracker import MLflowTracker


class TrainingPipeline:
    """Production training pipeline for Factory Guard AI."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline."""
        self.config = ConfigLoader(config_path)
        self.logger = Logger()
        self.tracker = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data using DataLoader."""
        loader = DataLoader(backend="pandas")
        df = loader.load_csv(data_path)
        
        # Preprocessing
        df = loader.handle_missing_values(df, strategy="mean")
        df = loader.remove_duplicates(df)
        
        return df
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features."""
        preprocessor = Preprocessor(scaling_method="standardscaler")
        X_scaled = preprocessor.fit_transform(X)
        
        # Save scaler for later use
        joblib.dump(preprocessor.scaler, "models/scaler.pkl")
        self.logger.info("Scaler saved to models/scaler.pkl")
        
        return X_scaled
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series,
             model_type: str = "xgboost") -> ModelTrainer:
        """Train model."""
        # Get model parameters, use defaults if not in config
        params = self.config.get(f"models.{model_type}", {})
        if params is None:
            params = {}
        
        self.logger.info(f"Training {model_type} model with parameters: {params}")
        
        trainer = ModelTrainer(model_type=model_type, **params)
        trainer.train(X_train, y_train, X_val, y_val)
        
        return trainer
    
    def evaluate(self, trainer: ModelTrainer, X_test: pd.DataFrame, 
                y_test: pd.Series) -> dict:
        """Evaluate model."""
        metrics = trainer.evaluate(X_test, y_test)
        
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def run(self, data_path: str = "data/raw/sample_data.csv",
           model_type: str = "xgboost", use_mlflow: bool = False):
        """Execute full training pipeline."""
        self.logger.info("Starting training pipeline...")
        
        # Initialize MLflow if requested
        if use_mlflow:
            self.tracker = MLflowTracker(
                experiment_name=self.config.get("mlflow.experiment_name", "Factory_Guard")
            )
            self.tracker.start_run(run_name=model_type)
        
        try:
            # Load data
            self.logger.info("Loading data...")
            df = self.load_data(data_path)
            
            # Separate features and target
            X = df.drop('target', axis=1, errors='ignore')
            y = df.get('target', pd.Series(np.random.randint(0, 2, len(df))))
            
            # Train-test split
            self.logger.info("Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
            )
            
            # Preprocess
            self.logger.info("Preprocessing features...")
            X_train_scaled = self.preprocess_features(X_train)
            
            scaler = joblib.load("models/scaler.pkl")
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val), 
                columns=X_val.columns, 
                index=X_val.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
            
            # Train
            trainer = self.train(X_train_scaled, y_train, X_val_scaled, y_val, model_type)
            
            # Evaluate
            metrics = self.evaluate(trainer, X_test_scaled, y_test)
            
            # Save model
            model_path = f"models/{model_type}_model.pkl"
            trainer.save_model(model_path)
            
            # Log to MLflow
            if self.tracker:
                self.tracker.log_params({"model_type": model_type})
                self.tracker.log_metrics(metrics)
                self.tracker.log_model(trainer.model, model_type, flavor="sklearn")
                self.tracker.end_run()
            
            self.logger.info("Pipeline completed successfully!")
            return trainer, metrics
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            if self.tracker:
                self.tracker.end_run(status="FAILED")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Factory Guard AI Training Pipeline")
    parser.add_argument("--data", type=str, default="data/raw/sample_data.csv",
                       help="Path to training data")
    parser.add_argument("--model", type=str, default="xgboost",
                       choices=["logistic_regression", "xgboost", "lightgbm", "random_forest"],
                       help="Model type to train")
    parser.add_argument("--mlflow", action="store_true", help="Use MLflow tracking")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Config file path")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = TrainingPipeline(config_path=args.config)
    trainer, metrics = pipeline.run(
        data_path=args.data,
        model_type=args.model,
        use_mlflow=args.mlflow
    )


if __name__ == "__main__":
    main()
