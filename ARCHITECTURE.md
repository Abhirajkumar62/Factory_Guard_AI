# Project Architecture

## High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│         FACTORY GUARD AI - ML PIPELINE                  │
└─────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
    ┌────────┐       ┌─────────┐     ┌────────────┐
    │ DATA   │       │ FEATURES│     │  MODELS    │
    │ LAYER  │──────▶│ LAYER   │────▶│ LAYER      │
    └────────┘       └─────────┘     └────────────┘
         ▲
         │
    ┌────────┐       ┌──────────┐    ┌──────────┐
    │ CONFIG │       │ UTILS    │    │ MLOps    │
    │ LAYER  │───────│ LAYER    │────│ LAYER    │
    └────────┘       └──────────┘    └──────────┘
```

## Component Architecture

### 1. Data Layer (`src/data/`)
- **DataLoader**: Handles Pandas/Spark backend selection
  - Loads CSV files
  - Handles missing values (mean, median, drop)
  - Removes duplicates
  - Performs train-test splits
  
- **Preprocessor**: Data scaling and cleaning
  - StandardScaler, MinMaxScaler, RobustScaler
  - Outlier detection (IQR, ZScore)
  - Feature normalization

### 2. Features Layer (`src/features/`)
- **FeatureEngineer**: Advanced feature creation
  - Correlation-based feature selection
  - Statistical feature importance
  - Polynomial features
  - Interaction features
  - Vectorization support

### 3. Models Layer (`src/models/`)
- **ModelTrainer**: Unified training interface
  - Classical ML: Logistic Regression, SVM, Random Forest
  - Gradient Boosting: XGBoost, LightGBM
  - Deep Learning: TensorFlow/Keras
  - Model evaluation with multiple metrics
  - Model persistence (joblib/pickle)

### 4. Utils Layer (`src/utils/`)
- **ConfigLoader**: YAML/environment configuration
- **Logger**: Structured logging
- **MLflowTracker**: Experiment tracking and versioning

## Data Flow

```
RAW DATA
   ↓
LOAD (DataLoader)
   ↓
PREPROCESS (Preprocessor)
   ├─ Handle missing values
   ├─ Remove outliers
   └─ Scale features
   ↓
ENGINEER (FeatureEngineer)
   ├─ Select features
   ├─ Create interactions
   └─ Vectorize
   ↓
SPLIT (Train/Val/Test)
   ↓
TRAIN (ModelTrainer)
   ├─ Cross-validation
   ├─ Hyperparameter tuning
   └─ Track with MLflow
   ↓
EVALUATE
   ├─ Accuracy, Precision, Recall, F1
   ├─ ROC-AUC, Confusion Matrix
   └─ Feature Importance
   ↓
SAVE MODEL + SCALER
   ↓
DEPLOY (predict.py)
```

## File Organization

```
src/
├── __init__.py              # Package initialization
├── data/
│   ├── __init__.py
│   ├── loader.py           # DataLoader class
│   └── preprocessing.py    # Preprocessor class
├── features/
│   ├── __init__.py
│   └── engineer.py         # FeatureEngineer class
├── models/
│   ├── __init__.py
│   └── trainer.py          # ModelTrainer class
└── utils/
    ├── __init__.py
    ├── config.py           # ConfigLoader, Logger
    └── mlflow_tracker.py   # MLflow integration
```

## Technology Stack

### Data Processing
- **Pandas** (v2.0+): DataFrame operations, data wrangling
- **NumPy** (v1.24+): Vectorized operations, array processing
- **PySpark** (v3.5+): Distributed computing for big data

### Classical ML
- **Scikit-Learn** (v1.3+): Regression, classification, preprocessing
- **Joblib**: Efficient model serialization

### Gradient Boosting
- **XGBoost** (v2.0+): Tree-based gradient boosting
- **LightGBM** (v4.0+): Fast and memory-efficient boosting

### Deep Learning
- **TensorFlow** (v2.14+): Neural network framework
- **Keras** (v2.14+): High-level API for deep learning

### MLOps
- **MLflow** (v2.10+): Experiment tracking and model registry
- **WandB**: Optional alternative for experiment tracking

### Development Tools
- **Jupyter**: Interactive notebook environment
- **pytest**: Unit testing framework
- **Black**: Code formatting
- **Flake8**: Code linting

## Configuration Management

Configuration stored in `config/config.yaml`:

```yaml
project:
  name: "Factory Guard AI"
  version: "0.1.0"

data:
  raw_path: "./data/raw"
  processed_path: "./data/processed"

models:
  xgboost:
    learning_rate: 0.1
    max_depth: 6
    n_estimators: 100

training:
  cross_validation_folds: 5
  optimizer: "adam"

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "Factory_Guard_Experiments"
```

## Scalability

### Small Data (<5GB)
Use in-memory processing with Pandas:
```python
loader = DataLoader(backend="pandas")
df = loader.load_csv("small_data.csv")
```

### Large Data (>5GB)
Use Spark distributed processing:
```python
loader = DataLoader(backend="spark")
spark_df = loader.load_csv("large_data.csv")
```

Spark automatically:
- Partitions data across cluster
- Handles out-of-core processing
- Optimizes operations

## Performance Characteristics

| Operation | Pandas | Spark | Time |
|-----------|--------|-------|------|
| Load 1GB CSV | ✓ | ✓ | <1s / <5s |
| Load 100GB CSV | ✗ | ✓ | <30s |
| Feature engineering | ✓ (fast) | ✓ | Immediate |
| Model training | ✓ | ✓ | ~1-10s |
| Hyperparameter tuning | ✓ (20 iterations) | ✓ | ~5-60s |

## Extension Points

### Add new data source
```python
# src/data/sources/
class PostgresLoader(DataLoader):
    def load_from_db(self, query):
        # Implementation
```

### Add new preprocessing method
```python
# src/data/preprocessing.py
def robust_feature_selection(self, X, y):
    # Implementation
```

### Add new model
```python
# src/models/trainer.py
elif model_type == "catboost":
    self.model = CatBoostClassifier(**params)
```

## Production Deployment

1. **Environment Setup**
   - Docker containerization
   - Environment variables
   - Secret management

2. **API Layer** (Flask/FastAPI)
   ```python
   @app.post("/predict")
   def predict(data: InputData):
       return predictor.predict(data.to_df())
   ```

3. **Monitoring**
   - Model drift detection
   - Performance metrics
   - Latency tracking
   - Error logging

4. **Versioning**
   - Model versioning in MLflow
   - A/B testing support
   - Rollback capability

## Best Practices Implemented

✅ **Modularity**: Separate concerns into independent modules
✅ **Reproducibility**: Configuration-driven, seed management
✅ **Testing**: Unit tests for data and feature modules
✅ **Documentation**: Comprehensive README and examples
✅ **Logging**: Structured logging throughout pipeline
✅ **Error Handling**: Graceful error handling and logging
✅ **Type Hints**: Type hints for better code clarity
✅ **Version Control**: .gitignore, branch strategy
✅ **CI/CD**: GitHub Actions workflow
✅ **MLOps**: MLflow integration for experiment tracking
✅ **Containerization**: Docker support for reproducible environments
✅ **Code Quality**: Linting and formatting with Black/Flake8
