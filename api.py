"""
Production API for Real-Time Anomaly Detection and Predictive Maintenance
Handles requests from 500 robotic arms with continuous monitoring.
"""

from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.data.sensor_pipeline import SensorDataPipeline
from src.models.failure_prediction import FailurePredictionEngine
from src.utils.config import Logger

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline and prediction engine
try:
    sensor_pipeline = SensorDataPipeline(n_arms=500)
    prediction_engine = FailurePredictionEngine(
        model_path="models/xgboost_model.pkl",
        scaler_path="models/scaler.pkl",
        failure_threshold=0.5
    )
    logger.info("✅ Prediction system initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize: {str(e)}")
    sensor_pipeline = None
    prediction_engine = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Factory Guard AI - Predictive Maintenance',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': prediction_engine.model is not None,
        'arms_monitored': sensor_pipeline.n_arms if sensor_pipeline else 0
    }), 200


@app.route('/predict/arm', methods=['POST'])
def predict_arm_failure():
    """
    Predict failure for a single robotic arm.
    
    Expected JSON:
    {
        "arm_id": "ARM_001",
        "temperature": 98.5,
        "pressure": 1013.0,
        "vibration": 2.5,
        "humidity": 55.0,
        "power_consumption": 510.0
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['arm_id', 'temperature', 'pressure', 'vibration', 'humidity', 'power_consumption']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Ingest sensor data
        sensor_data = sensor_pipeline.ingest_sensor_data(
            arm_id=data['arm_id'],
            temperature=data['temperature'],
            pressure=data['pressure'],
            vibration=data['vibration'],
            humidity=data['humidity'],
            power_consumption=data['power_consumption']
        )
        
        # Preprocess features
        features = sensor_pipeline.preprocess_features(sensor_data)
        
        # Get trend analysis
        trends = sensor_pipeline.calculate_trend_features(data['arm_id'])
        
        # Predict failure
        health_assessment = prediction_engine.analyze_arm_health(
            arm_id=data['arm_id'],
            current_features=features,
            trends=trends
        )
        
        return jsonify(health_assessment), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch_fleet():
    """
    Predict failures for multiple arms at once.
    
    Expected JSON:
    {
        "arms": [
            {
                "arm_id": "ARM_001",
                "temperature": 98.5,
                "pressure": 1013.0,
                "vibration": 2.5,
                "humidity": 55.0,
                "power_consumption": 510.0
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        arms = data.get('arms', [])
        
        if not arms:
            return jsonify({'error': 'No arms provided'}), 400
        
        fleet_data = []
        
        # Process each arm
        for arm in arms:
            sensor_data = sensor_pipeline.ingest_sensor_data(
                arm_id=arm['arm_id'],
                temperature=arm['temperature'],
                pressure=arm['pressure'],
                vibration=arm['vibration'],
                humidity=arm['humidity'],
                power_consumption=arm['power_consumption']
            )
            
            features = sensor_pipeline.preprocess_features(sensor_data)
            trends = sensor_pipeline.calculate_trend_features(arm['arm_id'])
            
            fleet_data.append({
                'arm_id': arm['arm_id'],
                'features': features,
                'trends': trends
            })
        
        # Batch predict
        assessments = prediction_engine.batch_predict_fleet(fleet_data)
        
        # Get alerts
        alerts = prediction_engine.get_failure_alerts(assessments)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'total_arms_processed': len(arms),
            'arms_with_alerts': len(alerts),
            'assessments': assessments,
            'critical_alerts': alerts
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/report/maintenance', methods=['GET'])
def get_maintenance_report():
    """Get comprehensive maintenance report for entire fleet."""
    try:
        # Get all health snapshots
        snapshots = sensor_pipeline.get_all_health_snapshot()
        
        if not snapshots:
            return jsonify({'warning': 'No data available yet'}), 200
        
        # Build fleet data for assessment
        fleet_data = []
        for arm_id, snapshot in snapshots.items():
            readings = snapshot['current_readings']
            sensor_data = {
                'temperature': readings['temperature'],
                'pressure': readings['pressure'],
                'vibration': readings['vibration'],
                'humidity': readings['humidity'],
                'power_consumption': readings['power_consumption']
            }
            features = sensor_pipeline.preprocess_features(sensor_data)
            
            fleet_data.append({
                'arm_id': arm_id,
                'features': features,
                'trends': snapshot['trends']
            })
        
        # Generate assessments and report
        assessments = prediction_engine.batch_predict_fleet(fleet_data)
        report = prediction_engine.generate_maintenance_report(assessments)
        
        return jsonify(report), 200
    
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard/alerts', methods=['GET'])
def get_alerts_dashboard():
    """Get current critical alerts for dashboard display."""
    try:
        snapshots = sensor_pipeline.get_all_health_snapshot()
        
        if not snapshots:
            return jsonify({'alerts': [], 'timestamp': datetime.now().isoformat()}), 200
        
        # Build fleet data
        fleet_data = []
        for arm_id, snapshot in snapshots.items():
            readings = snapshot['current_readings']
            sensor_data = {
                'temperature': readings['temperature'],
                'pressure': readings['pressure'],
                'vibration': readings['vibration'],
                'humidity': readings['humidity'],
                'power_consumption': readings['power_consumption']
            }
            features = sensor_pipeline.preprocess_features(sensor_data)
            
            fleet_data.append({
                'arm_id': arm_id,
                'features': features,
                'trends': snapshot['trends']
            })
        
        # Get alerts
        assessments = prediction_engine.batch_predict_fleet(fleet_data)
        alerts = prediction_engine.get_failure_alerts(assessments)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(alerts),
            'critical_alerts': [
                {
                    'arm_id': a['arm_id'],
                    'health_score': a['health_score'],
                    'failure_probability': a['failure_probability'],
                    'priority': a['maintenance_priority'],
                    'suggested_action': f"Schedule maintenance within {a['predicted_failure_window']['estimated_days']:.1f} days"
                }
                for a in alerts[:20]  # Top 20
            ]
        }), 200
    
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics."""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'arms_monitored': sensor_pipeline.n_arms,
        'model_status': 'ready' if prediction_engine.model else 'not_loaded',
        'failure_threshold': prediction_engine.failure_threshold,
        'api_version': '1.0.0'
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("🚀 Starting Factory Guard AI Predictive Maintenance API")
    logger.info("📊 Monitoring 500 robotic arms for catastrophic failures")
    logger.info("🔮 Predicting failures 24+ hours in advance")
    app.run(host='0.0.0.0', port=5001, debug=False)
