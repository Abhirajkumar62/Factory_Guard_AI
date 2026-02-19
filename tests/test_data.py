"""
Tests for data module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessing import Preprocessor


class TestDataLoader:
    """Test DataLoader class."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        csv_path = tmp_path / "sample.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_load_csv(self, sample_csv):
        """Test loading CSV file."""
        loader = DataLoader(backend="pandas")
        df = loader.load_csv(str(sample_csv))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert list(df.columns) == ["feature1", "feature2", "target"]

    def test_handle_missing_values(self):
        """Test missing value handling."""
        loader = DataLoader()

        df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, np.nan, np.nan, 8]})

        df_filled = loader.handle_missing_values(df, strategy="mean")

        assert df_filled["A"].notna().all()
        assert df_filled["B"].notna().all()

    def test_remove_duplicates(self):
        """Test duplicate removal."""
        loader = DataLoader()

        df = pd.DataFrame({"A": [1, 2, 2, 3], "B": [4, 5, 5, 6]})

        df_dedup = loader.remove_duplicates(df)

        assert len(df_dedup) == 3


class TestPreprocessor:
    """Test Preprocessor class."""

    def test_fit_transform(self):
        """Test scaling transformation."""
        preprocessor = Preprocessor(scaling_method="standardscaler")

        X = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})

        X_scaled = preprocessor.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert np.allclose(X_scaled.mean(), 0, atol=1e-10)

    def test_remove_outliers(self):
        """Test outlier removal."""
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 100], "B": [10, 20, 30, 40, 50]}  # 100 is an outlier
        )

        df_no_outliers = Preprocessor.remove_outliers(df, method="iqr", threshold=1.5)

        assert len(df_no_outliers) < len(df)
