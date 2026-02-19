"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample feature and target data."""
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    def test_create_polynomial_features(self, sample_data):
        """Test polynomial feature creation."""
        X, _ = sample_data
        engineer = FeatureEngineer()

        X_poly = engineer.create_polynomial_features(X, degree=2)

        # degree 2 polynomial with 3 features should have at least 6 features
        assert X_poly.shape[1] >= 6

    def test_select_features_by_correlation(self, sample_data):
        """Test feature selection by correlation."""
        X, y = sample_data
        engineer = FeatureEngineer()

        # Create a feature highly correlated with target
        X["correlated"] = y.values + np.random.randn(100) * 0.1

        X_selected = engineer.select_features_by_correlation(X, y, threshold=0.1)

        assert "correlated" in X_selected.columns

    def test_create_interaction_features(self, sample_data):
        """Test interaction feature creation."""
        X, _ = sample_data
        engineer = FeatureEngineer()

        pairs = [("feature1", "feature2")]
        X_interaction = engineer.create_interaction_features(X, feature_pairs=pairs)

        assert "feature1_x_feature2" in X_interaction.columns
