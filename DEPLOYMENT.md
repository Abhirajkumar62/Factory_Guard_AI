# Deployment Guide

## Local Development

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training
```bash
# Without MLflow
python train.py

# With MLflow tracking
python train.py --mlflow --model xgboost

# Custom data path
python train.py --data data/raw/my_data.csv --model lightgbm
```

### 3. Make Predictions
```bash
python predict.py
```

### 4. View MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

## Docker Deployment

### Build Image
```bash
docker build -t factory-guard-ai:latest .
```

### Run Container
```bash
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  factory-guard-ai:latest
```

### Docker Compose
```bash
docker-compose up -d
```

## Production Deployment

### AWS SageMaker
1. Package model as SageMaker inference container
2. Deploy using SageMaker endpoints
3. Enable auto-scaling based on traffic

### Kubernetes
```bash
# Create namespace
kubectl create namespace factory-guard

# Deploy using Helm
helm install factory-guard ./helm-chart -n factory-guard
```

### Cloud Run (GCP)
```bash
gcloud run deploy factory-guard-ai \
  --source . \
  --platform managed \
  --region us-central1
```

### Azure ML
```bash
# Create deployment
az ml model deploy \
  --model-id <model-id> \
  --deployment-name factory-guard-prod
```

## Monitoring

### Health Checks
```bash
curl http://localhost:5000/health
```

### Metrics
```bash
curl http://localhost:5000/metrics
```

### Logs
```bash
docker logs <container-id>
```

## Security Best Practices

1. **API Keys**: Use environment variables for sensitive data
2. **HTTPS**: Always use HTTPS in production
3. **Input Validation**: Validate all incoming requests
4. **Rate Limiting**: Implement rate limiting per API key
5. **Model Versioning**: Always version models in MLflow
6. **Access Control**: Restrict model artifact access
7. **Monitoring**: Log all predictions and anomalies
8. **Updates**: Plan regular model retraining

## Rollback Procedure

1. Identify degraded model performance in monitoring
2. Stop current deployment
3. Deploy previous model version from MLflow
4. Verify predictions on test data
5. Investigate root cause
6. Retrain and deploy new version

## Performance Optimization

- **Batch Predictions**: Use batch endpoint for bulk requests
- **Caching**: Cache scaler and model in memory
- **GPU**: Enable GPU for deep learning models
- **Quantization**: Use model quantization for faster inference
