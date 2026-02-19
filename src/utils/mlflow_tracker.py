"""
MLflow integration module.
Handles experiment tracking, model logging, and registry management.
"""

import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
from src.utils.config import Logger


class MLflowTracker:
    """MLflow integration for experiment tracking and model management."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "Factory_Guard_Experiments",
    ):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.logger = Logger()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.logger.info(f"Initialized MLflow tracker: {tracking_uri}")

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Name of the run
            tags: Dictionary of tags for the run
        """
        mlflow.start_run(run_name=run_name)

        if tags:
            mlflow.set_tags(tags)

        self.logger.info(f"Started MLflow run: {run_name}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)
        self.logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
        self.logger.info(f"Logged {len(metrics)} metrics")

    def log_model(self, model, model_name: str, flavor: str = "sklearn"):
        """
        Log model to MLflow.

        Args:
            model: Trained model
            model_name: Name for the model
            flavor: Model flavor ("sklearn", "tensorflow", "xgboost", etc.)
        """
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif flavor == "tensorflow":
            mlflow.tensorflow.log_model(model, model_name)
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.log_model(model, model_name)

        self.logger.info(f"Logged model: {model_name}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log artifact (file or directory).

        Args:
            local_path: Path to local file/directory
            artifact_path: Remote path in artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        self.logger.info(f"Logged artifact: {local_path}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status ("FINISHED" or "FAILED")
        """
        mlflow.end_run(status=status)
        self.logger.info(f"Ended MLflow run with status: {status}")

    @staticmethod
    def get_best_run(experiment_name: str, metric_name: str, ascending: bool = False):
        """
        Get the best run from an experiment.

        Args:
            experiment_name: Name of the experiment
            metric_name: Name of the metric to optimize
            ascending: If True, minimize metric, else maximize

        Returns:
            Best run object
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            return None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if ascending:
            best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmin()]
        else:
            best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmax()]

        return best_run
