"""
Model training and evaluation module.
Supports Scikit-Learn, XGBoost, LightGBM, and TensorFlow/Keras.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
from src.utils.config import Logger


class ModelTrainer:
    """Train and evaluate machine learning models."""
    
    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        """
        Initialize ModelTrainer.
        
        Args:
            model_type: Type of model ("logistic_regression", "xgboost", "lightgbm", "neural_network")
            **kwargs: Model parameters
        """
        self.model_type = model_type
        self.model = None
        self.logger = Logger()
        self.history = None
        
        self._initialize_model(model_type, kwargs)
    
    def _initialize_model(self, model_type: str, params: Dict[str, Any]):
        """Initialize model based on type."""
        if model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(**params)
        
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**params)
        
        elif model_type == "svm":
            from sklearn.svm import SVC
            self.model = SVC(**params)
        
        elif model_type == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(**params)
            except ImportError:
                self.logger.error("XGBoost not installed")
                raise
        
        elif model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(**params)
            except ImportError:
                self.logger.error("LightGBM not installed")
                raise
        
        elif model_type == "neural_network":
            self._initialize_neural_network(params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.logger.info(f"Initialized {model_type} model")
    
    def _initialize_neural_network(self, params: Dict[str, Any]):
        """Initialize neural network model."""
        try:
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Input
            from tensorflow.keras.optimizers import Adam
            
            architecture = params.get("architecture", "dense")
            
            self.model = Sequential([
                Dense(128, activation='relu', input_shape=(params.get("input_dim", 10),)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.logger.info("Initialized neural network model")
        
        except ImportError:
            self.logger.error("TensorFlow not installed")
            raise
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training history/results
        """
        self.logger.info(f"Training {self.model_type} model")
        
        if self.model_type == "neural_network":
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=kwargs.get("epochs", 50),
                batch_size=kwargs.get("batch_size", 32),
                verbose=kwargs.get("verbose", 1)
            )
            
            return {
                "loss": self.history.history.get("loss", [])[-1],
                "accuracy": self.history.history.get("accuracy", [])[-1]
            }
        
        else:
            self.model.fit(X_train, y_train)
            
            train_score = self.model.score(X_train, y_train)
            self.logger.info(f"Training accuracy: {train_score:.4f}")
            
            return {"train_score": train_score}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (if available)."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            self.logger.warning("Model does not support predict_proba")
            return None
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                    f1_score, roc_auc_score, confusion_matrix)
        
        y_pred = self.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # Add ROC-AUC if available
        if hasattr(self.model, "predict_proba"):
            y_proba = self.predict_proba(X_test)
            if y_proba is not None and len(np.unique(y_test)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == "neural_network":
            self.model.save(filepath)
        else:
            joblib.dump(self.model, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        if self.model_type == "neural_network":
            from tensorflow.keras.models import load_model as tf_load_model
            self.model = tf_load_model(filepath)
        else:
            self.model = joblib.load(filepath)
        
        self.logger.info(f"Model loaded from {filepath}")
