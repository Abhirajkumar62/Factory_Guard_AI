# API Documentation

## Overview
Factory Guard AI provides production-ready endpoints for model inference and monitoring.

## Endpoints

### POST /predict
Make predictions on new data.

**Request:**
```json
{
  "features": {
    "temperature": 98.5,
    "pressure": 1013.0,
    "vibration": 2.5,
    "humidity": 55.0,
    "power_consumption": 510.0
  }
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.95,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

### GET /metrics
Get model performance metrics.

**Response:**
```json
{
  "accuracy": 0.95,
  "precision": 0.92,
  "recall": 0.88,
  "f1": 0.90
}
```

## Authentication
All endpoints require API key authentication via header:
```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting
- 1000 requests per hour per API key
- 100 requests per minute for /predict

## Error Handling
- 400: Bad Request
- 401: Unauthorized
- 429: Too Many Requests
- 500: Internal Server Error
