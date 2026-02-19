"""Create sample data for Factory Guard AI project."""
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Create sample dataset
df = pd.DataFrame({
    'temperature': np.random.normal(98.6, 5, n_samples),
    'pressure': np.random.normal(1013.25, 10, n_samples),
    'vibration': np.random.exponential(2, n_samples),
    'humidity': np.random.uniform(30, 80, n_samples),
    'power_consumption': np.random.normal(500, 100, n_samples),
})

# Create target variable (anomaly detection)
# Anomaly when temperature > 110 or vibration > 8
df['target'] = ((df['temperature'] > 110) | (df['vibration'] > 8)).astype(int)

# Save to CSV
df.to_csv('data/raw/sample_data.csv', index=False)

print(f"Created sample data: {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")
print(f"Saved to: data/raw/sample_data.csv")
