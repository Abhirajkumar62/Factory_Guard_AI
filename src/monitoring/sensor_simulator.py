"""
Sensor Data Simulator for Testing 24-Hour Predictive Maintenance
Mimics real-world sensor readings from 500 robotic arms with failure patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time
from typing import List, Dict
import random

from src.utils.config import Logger


class SensorSimulator:
    """Simulates sensor readings from robotic arms with realistic failure patterns."""

    def __init__(self, n_arms: int = 500, api_endpoint: str = "http://localhost:5001"):
        """
        Initialize sensor simulator.

        Args:
            n_arms: Number of robotic arms to simulate
            api_endpoint: API endpoint for predictions
        """
        self.n_arms = n_arms
        self.api_endpoint = api_endpoint
        self.logger = Logger()

        # Baseline sensor values for healthy arms
        self.baseline = {
            "temperature": 95.0,  # °C
            "pressure": 1013.0,  # Pa
            "vibration": 2.0,  # mm/s
            "humidity": 50.0,  # %
            "power_consumption": 500.0,  # W
        }

        # Arms that will fail in next 24 hours (for testing)
        self.failure_arms = set(
            random.sample(range(n_arms), max(1, n_arms // 100))
        )  # ~1% will fail

        self.logger.info(f"Initialized simulator for {n_arms} arms")
        self.logger.info(
            f"⚠️ {len(self.failure_arms)} arms scheduled to fail in test scenario"
        )

    def generate_healthy_readings(self) -> Dict:
        """Generate sensor readings for healthy arm."""
        return {
            "temperature": self.baseline["temperature"] + np.random.normal(0, 2),
            "pressure": self.baseline["pressure"] + np.random.normal(0, 5),
            "vibration": self.baseline["vibration"] + np.random.normal(0, 0.3),
            "humidity": self.baseline["humidity"] + np.random.normal(0, 5),
            "power_consumption": self.baseline["power_consumption"]
            + np.random.normal(0, 20),
        }

    def generate_degrading_readings(self, time_step: float) -> Dict:
        """
        Generate sensor readings for arm approaching failure.

        Args:
            time_step: Position in degradation timeline (0.0 to 1.0)
                      0.0 = healthy, 1.0 = failure

        Returns:
            Sensor readings showing degradation
        """
        # Gradual increase over 24 hours
        temp_increase = 15 * time_step  # Up to 15°C increase
        vibration_increase = 4 * (time_step**1.5)  # Accelerating vibration
        pressure_increase = 100 * (time_step**2)  # Exponential pressure spike
        power_increase = 200 * time_step  # Rising power consumption

        return {
            "temperature": self.baseline["temperature"]
            + temp_increase
            + np.random.normal(0, 1),
            "pressure": self.baseline["pressure"]
            + pressure_increase
            + np.random.normal(0, 10),
            "vibration": self.baseline["vibration"]
            + vibration_increase
            + np.random.normal(0, 0.2),
            "humidity": self.baseline["humidity"] + np.random.normal(0, 5),
            "power_consumption": self.baseline["power_consumption"]
            + power_increase
            + np.random.normal(0, 30),
        }

    def generate_arm_readings(
        self, arm_index: int, time_since_start: float = 0
    ) -> Dict:
        """
        Generate readings for a single arm.

        Args:
            arm_index: Index of the arm (0 to n_arms-1)
            time_since_start: Hours since simulation start

        Returns:
            Dictionary with sensor readings
        """
        arm_id = f"ARM_{arm_index:03d}"

        if arm_index in self.failure_arms:
            # This arm degrades over 24 hours
            time_step = (time_since_start % 24) / 24.0  # Position in 24-hour cycle
            readings = self.generate_degrading_readings(time_step)
        else:
            # Normal operation
            readings = self.generate_healthy_readings()

        readings["arm_id"] = arm_id
        return readings

    def generate_fleet_readings(
        self, time_since_start: float = 0, sample_size: int = None
    ) -> List[Dict]:
        """
        Generate sensor readings for entire fleet or sample.

        Args:
            time_since_start: Hours since simulation start
            sample_size: Number of arms to read (None = all 500)

        Returns:
            List of sensor readings
        """
        if sample_size is None:
            sample_size = self.n_arms

        arm_indices = random.sample(range(self.n_arms), min(sample_size, self.n_arms))

        readings = []
        for idx in arm_indices:
            readings.append(self.generate_arm_readings(idx, time_since_start))

        return readings

    def send_prediction_request(self, readings: Dict) -> Dict:
        """
        Send sensor readings to API for prediction.

        Args:
            readings: Sensor data for one arm

        Returns:
            Prediction response
        """
        try:
            response = requests.post(
                f"{self.api_endpoint}/predict/arm", json=readings, timeout=5
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            return None

    def send_batch_prediction(self, all_readings: List[Dict]) -> Dict:
        """
        Send batch of readings for processing.

        Args:
            all_readings: List of sensor readings

        Returns:
            Batch prediction response
        """
        try:
            response = requests.post(
                f"{self.api_endpoint}/predict/batch",
                json={"arms": all_readings},
                timeout=10,
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Batch API request failed: {str(e)}")
            return None

    def run_simulation(
        self,
        duration_hours: int = 24,
        interval_minutes: int = 5,
        sample_size: int = 100,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Run continuous simulation of sensor data.

        Args:
            duration_hours: Length of simulation
            interval_minutes: Time between readings
            sample_size: Number of arms to monitor per cycle
            verbose: Print progress

        Returns:
            List of all predictions
        """
        predictions = []
        elapsed_hours = 0

        self.logger.info(f"🚀 Starting {duration_hours}-hour simulation")
        self.logger.info(
            f"📊 Sampling {sample_size} arms every {interval_minutes} minutes"
        )

        while elapsed_hours < duration_hours:
            # Generate readings for sample of arms
            readings = self.generate_fleet_readings(
                time_since_start=elapsed_hours, sample_size=sample_size
            )

            # Get predictions
            result = self.send_batch_prediction(readings)

            if result:
                predictions.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_hours": elapsed_hours,
                        "result": result,
                    }
                )

                # Log alerts
                alerts = result.get("critical_alerts", [])
                if alerts and verbose:
                    self.logger.warning(
                        f"🚨 {len(alerts)} ALERTS at hour {elapsed_hours:.1f}:"
                    )
                    for alert in alerts[:3]:
                        self.logger.warning(
                            f"   - {alert['arm_id']}: "
                            f"Health {alert['health_score']:.1f}%, "
                            f"Failure Prob {alert['failure_probability']:.2%}, "
                            f"Priority: {alert['priority']}"
                        )

                if verbose:
                    self.logger.info(
                        f"✅ Hour {elapsed_hours:.1f}: {result['total_arms_processed']} arms processed, "
                        f"{result['arms_with_alerts']} with alerts"
                    )

            # Wait for next cycle
            time.sleep(interval_minutes * 60)
            elapsed_hours += interval_minutes / 60

        self.logger.info(
            f"✅ Simulation complete. {len(predictions)} prediction cycles recorded"
        )
        return predictions

    def run_quick_test(self) -> None:
        """Run quick test with 10 readings."""
        self.logger.info("🧪 Running quick test with 10 arms...")

        for i in range(10):
            readings = self.generate_fleet_readings(sample_size=10)
            result = self.send_batch_prediction(readings)

            if result:
                alerts = result.get("critical_alerts", [])
                self.logger.info(f"Cycle {i+1}: {len(alerts)} alerts")

                if alerts:
                    for alert in alerts[:2]:
                        self.logger.warning(
                            f"  ⚠️ {alert['arm_id']}: "
                            f"{alert['failure_probability']:.1%} failure probability"
                        )

            time.sleep(2)

        self.logger.info("✅ Quick test complete")


if __name__ == "__main__":
    # Test the simulator
    simulator = SensorSimulator(n_arms=500, api_endpoint="http://localhost:5001")

    # Run quick test first
    simulator.run_quick_test()
