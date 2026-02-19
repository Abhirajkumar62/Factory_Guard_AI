"""
Predictive Maintenance Engine for 24-Hour Failure Prediction
Analyzes sensor patterns and predicts catastrophic failures before they occur.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import joblib
from pathlib import Path

from src.utils.config import Logger


class FailurePredictionEngine:
    """Engine for predicting equipment failures 24 hours in advance."""

    def __init__(
        self,
        model_path: str = "models/xgboost_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        failure_threshold: float = 0.5,
    ):
        """
        Initialize prediction engine.

        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            failure_threshold: Probability threshold for failure alert
        """
        self.logger = Logger()
        self.model = joblib.load(model_path) if Path(model_path).exists() else None
        self.scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None
        self.failure_threshold = failure_threshold
        self.predictions_history = {}

        self.logger.info(
            f"Initialized prediction engine (threshold: {failure_threshold})"
        )

    def predict_failure_probability(self, features: np.ndarray) -> Tuple[float, str]:
        """
        Predict probability of failure in next 24 hours.

        Args:
            features: Preprocessed feature vector

        Returns:
            Tuple of (probability, risk_level)
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return 0.0, "unknown"

        try:
            # Get prediction probability
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)[0]
                failure_prob = proba[1]  # Probability of failure class
            else:
                failure_prob = float(self.model.predict(features)[0])

            # Determine risk level
            if failure_prob < 0.2:
                risk_level = "low"
            elif failure_prob < 0.5:
                risk_level = "medium"
            elif failure_prob < 0.8:
                risk_level = "high"
            else:
                risk_level = "critical"

            return failure_prob, risk_level

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return 0.0, "error"

    def analyze_arm_health(
        self, arm_id: str, current_features: np.ndarray, trends: Dict = None
    ) -> Dict:
        """
        Comprehensive health analysis for a robotic arm.

        Args:
            arm_id: Robotic arm identifier
            current_features: Current sensor readings (preprocessed)
            trends: Historical trend metrics

        Returns:
            Dictionary with health assessment
        """
        failure_prob, risk_level = self.predict_failure_probability(current_features)

        # Analyze trends
        trend_health = self._assess_trends(trends or {})

        # Overall health score (0-100)
        health_score = (1 - failure_prob) * 100

        health_assessment = {
            "arm_id": arm_id,
            "timestamp": datetime.now().isoformat(),
            "failure_probability": float(failure_prob),
            "risk_level": risk_level,
            "health_score": float(health_score),
            "alert_needed": bool(failure_prob > self.failure_threshold),
            "trend_analysis": trend_health,
            "maintenance_priority": self._calculate_priority(failure_prob),
            "predicted_failure_window": self._estimate_failure_window(failure_prob),
        }

        return health_assessment

    def _assess_trends(self, trends: Dict) -> Dict:
        """Assess health based on trends."""
        if not trends:
            return {"status": "no_data", "warnings": []}

        warnings = []

        # Check for concerning trends
        if trends.get("temp_trend", 0) > 5:
            warnings.append("Rapid temperature increase")

        if trends.get("vibration_trend", 0) > 2:
            warnings.append("Increasing vibration")

        if trends.get("pressure_trend", 0) > 50:
            warnings.append("Pressure spike detected")

        if trends.get("vibration_max", 0) > 8:
            warnings.append("High vibration levels")

        if trends.get("temp_std", 0) > 5:
            warnings.append("Temperature instability")

        return {
            "status": "warning" if warnings else "normal",
            "warnings": warnings,
            "trend_score": float(
                max(0, 1 - (len(warnings) * 0.15))
            ),  # Penalty for each warning
        }

    def _calculate_priority(self, failure_prob: float) -> str:
        """Calculate maintenance priority."""
        if failure_prob > 0.8:
            return "immediate"  # Schedule within 4 hours
        elif failure_prob > 0.5:
            return "urgent"  # Schedule within 12 hours
        elif failure_prob > 0.2:
            return "high"  # Schedule within 24 hours
        else:
            return "routine"  # Routine maintenance schedule

    def _estimate_failure_window(self, failure_prob: float) -> Dict:
        """Estimate when failure might occur."""
        if failure_prob < 0.2:
            return {"estimated_days": None, "confidence": "low"}

        # Rough estimate based on probability
        # Higher probability = sooner failure
        days_until_failure = max(0.5, 7 * (1 - failure_prob))  # 0.5 to 7 days

        return {
            "earliest": (datetime.now() + timedelta(hours=1)).isoformat(),
            "latest": (datetime.now() + timedelta(days=days_until_failure)).isoformat(),
            "estimated_days": days_until_failure,
            "confidence": "medium" if 0.5 <= failure_prob <= 0.8 else "high",
        }

    def batch_predict_fleet(self, fleet_data: List[Dict]) -> List[Dict]:
        """
        Predict health for entire fleet of robotic arms.

        Args:
            fleet_data: List of dictionaries with arm_id and features

        Returns:
            List of health assessments
        """
        assessments = []

        for arm_data in fleet_data:
            assessment = self.analyze_arm_health(
                arm_id=arm_data["arm_id"],
                current_features=arm_data["features"],
                trends=arm_data.get("trends"),
            )
            assessments.append(assessment)

        return assessments

    def get_failure_alerts(self, fleet_assessments: List[Dict]) -> List[Dict]:
        """
        Filter assessments to get only critical failure alerts.

        Args:
            fleet_assessments: List of health assessments

        Returns:
            List of arms needing immediate attention
        """
        alerts = [a for a in fleet_assessments if a["alert_needed"]]

        # Sort by priority
        priority_order = {"immediate": 0, "urgent": 1, "high": 2, "routine": 3}
        alerts.sort(key=lambda x: priority_order.get(x["maintenance_priority"], 4))

        return alerts

    def generate_maintenance_report(self, fleet_assessments: List[Dict]) -> Dict:
        """
        Generate comprehensive maintenance report for management.

        Args:
            fleet_assessments: List of health assessments

        Returns:
            Summary report
        """
        alerts = self.get_failure_alerts(fleet_assessments)

        report = {
            "report_time": datetime.now().isoformat(),
            "total_arms_monitored": len(fleet_assessments),
            "arms_with_alerts": len(alerts),
            "immediate_action_required": len(
                [a for a in alerts if a["maintenance_priority"] == "immediate"]
            ),
            "urgent_maintenance": len(
                [a for a in alerts if a["maintenance_priority"] == "urgent"]
            ),
            "estimated_downtime_avoided": f"${len(alerts) * 5_000_000}",  # $5M per arm
            "critical_alerts": [
                {
                    "arm_id": a["arm_id"],
                    "failure_probability": a["failure_probability"],
                    "priority": a["maintenance_priority"],
                    "failure_window": a["predicted_failure_window"],
                    "health_score": a["health_score"],
                }
                for a in alerts[:10]  # Top 10 alerts
            ],
            "average_fleet_health": float(
                np.mean([a["health_score"] for a in fleet_assessments])
            ),
        }

        return report
