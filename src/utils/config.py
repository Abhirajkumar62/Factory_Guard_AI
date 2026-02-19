"""
Configuration loader module.
Handles YAML and environment configuration management.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML and environment variables."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ConfigLoader.

        Args:
            config_path: Path to configuration YAML file
        """
        load_dotenv(".env")
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value by dotted key.

        Example:
            config.get("data.raw_path")
            config.get("models.baseline.params.max_iter")
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config


class Logger:
    """Simple logging utility."""

    @staticmethod
    def log(message: str, level: str = "INFO"):
        """Log message with timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    @staticmethod
    def info(message: str):
        """Log info level message."""
        Logger.log(message, "INFO")

    @staticmethod
    def warning(message: str):
        """Log warning level message."""
        Logger.log(message, "WARNING")

    @staticmethod
    def error(message: str):
        """Log error level message."""
        Logger.log(message, "ERROR")
