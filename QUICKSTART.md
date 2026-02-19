# Getting Started - Quick Reference

## Installation (5 minutes)

```bash
cd Factory_Guard_AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Hello World Example

Create `hello_world.py`:

```python
import sys
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.models.trainer import ModelTrainer
from src.utils.config import ConfigLoader, Logger

# Load configuration
config = ConfigLoader("config/config.yaml")
logger = Logger()

# Create sample data
import pandas as pd
import numpy as np

X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
y = pd.Series(np.random.randint(0, 2, 100))

logger.info("Starting training pipeline...")

# Train a model
trainer = ModelTrainer(model_type="logistic_regression", random_state=42)
trainer.train(X, y)

# Evaluate
metrics = trainer.evaluate(X, y)
print("Metrics:", metrics)

logger.info("Pipeline complete!")
```

Run it:
```bash
python hello_world.py
```

## Common Tasks

### Load Data
```python
from src.data.loader import DataLoader

loader = DataLoader(backend="pandas")
df = loader.load_csv("data/raw/data.csv")
```

### Preprocess Data
```python
from src.data.preprocessing import Preprocessor

preprocessor = Preprocessor(scaling_method="standardscaler")
X_scaled = preprocessor.fit_transform(X)
```

### Select Features
```python
from src.features.engineer import FeatureEngineer

engineer = FeatureEngineer()
X_selected = engineer.select_features_by_importance(X, y, n_features=10)
```

### Train Model
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(model_type="xgboost", n_estimators=100)
trainer.train(X_train, y_train)
metrics = trainer.evaluate(X_test, y_test)
```

### Track with MLflow
```python
from src.utils.mlflow_tracker import MLflowTracker

tracker = MLflowTracker()
tracker.start_run("my_experiment")
tracker.log_params({"learning_rate": 0.1})
tracker.log_metrics({"accuracy": 0.95})
tracker.end_run()
```

## Next Steps

1. Review `config/config.yaml` - customize for your project
2. Create notebooks in `notebooks/` for EDA
3. Move validated code to `src/`
4. Run `pytest tests/` to ensure quality
5. Use MLflow UI to track experiments
