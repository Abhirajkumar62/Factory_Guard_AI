"""
Prediction script for Factory Guard AI.
Production-ready inference script for making predictions on new data.
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Logger


class Predictor:
    """Production prediction service."""
    
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor."""
        self.logger = Logger()
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.logger.info("Model and scaler loaded successfully")
    
    def predict(self, X: pd.DataFrame) -> dict:
        """Make predictions."""
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }
    
    def predict_single(self, features: dict) -> dict:
        """Predict for a single sample."""
        df = pd.DataFrame([features])
        return self.predict(df)


# Example usage
if __name__ == "__main__":
    # Load predictor
    predictor = Predictor(
        model_path="models/xgboost_model.pkl",
        scaler_path="models/scaler.pkl"
    )
    
    # Example prediction
    sample_data = pd.DataFrame({
        'temperature': [98.5],
        'pressure': [1013.0],
        'vibration': [2.5],
        'humidity': [55.0],
        'power_consumption': [510.0]
    })
    
    result = predictor.predict(sample_data)
    print(f"Prediction: {result['predictions'][0]}")
    if result['probabilities'] is not None:
        print(f"Confidence: {result['probabilities'][0]}")
