"""
Feature engineering module.
Implements vectorization and feature selection methods.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Tuple
from src.utils.config import Logger


class FeatureEngineer:
    """Feature engineering and selection utilities."""

    def __init__(self, method: str = "correlation"):
        """
        Initialize FeatureEngineer.

        Args:
            method: "correlation", "rfe", or "mutual_info"
        """
        self.method = method
        self.logger = Logger()
        self.selected_features = None

    def create_polynomial_features(
        self, X: pd.DataFrame, degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features for non-linear relationships.

        Args:
            X: Input features
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features
        """
        self.logger.info(f"Creating polynomial features with degree={degree}")

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

    def select_features_by_correlation(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Select features based on correlation with target.

        Args:
            X: Features
            y: Target variable
            threshold: Correlation threshold

        Returns:
            DataFrame with selected features
        """
        self.logger.info(f"Selecting features by correlation (threshold={threshold})")

        correlations = X.corrwith(y).abs()
        selected = correlations[correlations > threshold].index.tolist()

        self.selected_features = selected
        self.logger.info(f"Selected {len(selected)} features")

        return X[selected]

    def select_features_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        method: str = "f_classif",
    ) -> pd.DataFrame:
        """
        Select top features using statistical tests.

        Args:
            X: Features
            y: Target variable
            n_features: Number of features to select
            method: "f_classif" or "mutual_info"

        Returns:
            DataFrame with top features
        """
        self.logger.info(f"Selecting top {n_features} features using {method}")

        if method == "f_classif":
            score_func = f_classif
        elif method == "mutual_info":
            score_func = mutual_info_classif
        else:
            raise ValueError("Method must be 'f_classif' or 'mutual_info'")

        selector = SelectKBest(score_func=score_func, k=min(n_features, len(X.columns)))
        X_selected = selector.fit_transform(X, y)

        selected_cols = X.columns[selector.get_support()].tolist()
        self.selected_features = selected_cols

        return pd.DataFrame(X_selected, columns=selected_cols)

    def create_interaction_features(
        self, X: pd.DataFrame, feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between specified pairs.

        Args:
            X: Input features
            feature_pairs: List of (col1, col2) tuples

        Returns:
            DataFrame with interaction features added
        """
        X_copy = X.copy()

        for col1, col2 in feature_pairs:
            if col1 in X.columns and col2 in X.columns:
                X_copy[f"{col1}_x_{col2}"] = X[col1] * X[col2]

        self.logger.info(f"Created {len(feature_pairs)} interaction features")
        return X_copy
