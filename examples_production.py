#!/usr/bin/env python3
"""
End-to-End Production Example
Demonstrates complete Factory Guard AI workflow for 500 robotic arms
"""

import requests
import json
from datetime import datetime
import time
from src.monitoring.sensor_simulator import SensorSimulator
from src.monitoring.alert_manager import AlertManager
from src.utils.config import Logger

logger = Logger()

# ============================================================================
# PRODUCTION EXAMPLE: 500 Robotic Arms Predictive Maintenance
# ============================================================================

def example_single_arm_prediction():
    """Example 1: Predict failure for a single arm"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Arm Prediction")
    print("="*70)
    
    API_ENDPOINT = "http://localhost:5001"
    
    # Simulate healthy arm
    healthy_arm = {
        "arm_id": "ARM_001",
        "temperature": 98.5,
        "pressure": 1013.0,
        "vibration": 2.1,
        "humidity": 55.0,
        "power_consumption": 510.0
    }
    
    print("\n📊 Healthy Arm:")
    print(f"  Temperature: {healthy_arm['temperature']}°C")
    print(f"  Vibration: {healthy_arm['vibration']} mm/s")
    print(f"  Pressure: {healthy_arm['pressure']} Pa")
    
    try:
        response = requests.post(
            f"{API_ENDPOINT}/predict/arm",
            json=healthy_arm,
            timeout=5
        )
        result = response.json()
        
        print(f"\n✅ Prediction Result:")
        print(f"  Failure Probability: {result['failure_probability']:.1%}")
        print(f"  Risk Level: {result['risk_level'].upper()}")
        print(f"  Health Score: {result['health_score']:.1f}/100")
        print(f"  Alert Needed: {result['alert_needed']}")
        print(f"  Maintenance Priority: {result['maintenance_priority'].upper()}")
        
    except Exception as e:
        logger.error(f"Failed to connect to API: {str(e)}")
        print("❌ Make sure API is running: python api.py")
        return
    
    # Simulate degrading arm
    print("\n\n📊 Degrading Arm (showing problems):")
    degrading_arm = {
        "arm_id": "ARM_250",
        "temperature": 115.2,    # ↑ Increasing
        "pressure": 1150.5,      # ↑ Rising
        "vibration": 6.8,        # ↑↑ Dangerous
        "humidity": 62.0,
        "power_consumption": 690.0  # ↑ High
    }
    
    print(f"  Temperature: {degrading_arm['temperature']}°C (+16.7°C! ⚠️)")
    print(f"  Vibration: {degrading_arm['vibration']} mm/s (3.2x normal! 🚨)")
    print(f"  Pressure: {degrading_arm['pressure']} Pa (elevated)")
    
    response = requests.post(
        f"{API_ENDPOINT}/predict/arm",
        json=degrading_arm,
        timeout=5
    )
    result = response.json()
    
    print(f"\n🚨 CRITICAL Prediction:")
    print(f"  Failure Probability: {result['failure_probability']:.1%}")
    print(f"  Risk Level: {result['risk_level'].upper()}")
    print(f"  Health Score: {result['health_score']:.1f}/100")
    print(f"  Alert Needed: {result['alert_needed']}")
    print(f"  Maintenance Priority: {result['maintenance_priority'].upper()}")
    if result['alert_needed']:
        print(f"  Failure Window: {result['predicted_failure_window']}")
        print(f"\n  RECOMMENDED ACTION: {result.get('maintenance_action', 'Urgent inspection required')}")


def example_batch_prediction():
    """Example 2: Predict for entire fleet (batch)"""
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Batch Fleet Prediction (50 arms)")
    print("="*70)
    
    API_ENDPOINT = "http://localhost:5001"
    
    print("\n🚀 Generating readings from 50 random arms...")
    simulator = SensorSimulator(n_arms=500)
    
    # Get 50 samples
    readings = simulator.generate_fleet_readings(sample_size=50)
    
    print(f"✅ Generated {len(readings)} sensor readings")
    
    try:
        response = requests.post(
            f"{API_ENDPOINT}/predict/batch",
            json={"arms": readings},
            timeout=10
        )
        result = response.json()
        
        print(f"\n📊 Batch Results:")
        print(f"  Total arms processed: {result['total_arms_processed']}")
        print(f"  Arms with alerts: {result['arms_with_alerts']}")
        
        if result['critical_alerts']:
            print(f"\n🚨 Critical Alerts ({len(result['critical_alerts'])}):")
            for alert in result['critical_alerts'][:5]:
                print(f"    {alert['arm_id']:>10} | "
                      f"Failure: {alert['failure_probability']:>6.1%} | "
                      f"Health: {alert['health_score']:>5.1f} | "
                      f"Priority: {alert['maintenance_priority'].upper()}")
        else:
            print("\n✅ No critical alerts - fleet is healthy!")
            
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")


def example_monitoring_workflow():
    """Example 3: Continuous monitoring workflow"""
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Continuous Monitoring Workflow (3 cycles)")
    print("="*70)
    
    API_ENDPOINT = "http://localhost:5001"
    simulator = SensorSimulator(n_arms=500)
    alert_mgr = AlertManager()
    
    print("\n📊 Starting 3-cycle monitoring simulation...")
    print("(In production, this would run continuously)")
    
    for cycle in range(1, 4):
        print(f"\n\n--- Monitoring Cycle {cycle} (Hour {cycle * 4}) ---")
        
        # Get 50 samples
        readings = simulator.generate_fleet_readings(
            time_since_start=cycle * 4,
            sample_size=50
        )
        
        # Batch predict
        try:
            response = requests.post(
                f"{API_ENDPOINT}/predict/batch",
                json={"arms": readings},
                timeout=10
            )
            result = response.json()
            
            print(f"✅ Processed {result['total_arms_processed']} arms")
            print(f"   Alerts: {result['arms_with_alerts']}")
            
            # Handle alerts
            if result['critical_alerts']:
                print(f"\n🚨 Critical alerts detected!")
                
                for alert_data in result['critical_alerts'][:3]:
                    # Create alert using alert manager
                    alert = alert_mgr.create_alert(
                        alert_data['arm_id'],
                        alert_data
                    )
                    
                    print(f"\n   📌 {alert['arm_id']}")
                    print(f"      Severity: {alert['severity'].upper()}")
                    print(f"      Failure Prob: {alert['failure_probability']:.1%}")
                    print(f"      Action: {alert['maintenance_action']}")
                    print(f"      Cost If Unattended: {alert['estimated_replacement_cost']}")
        
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {str(e)}")
            continue
        
        if cycle < 3:
            print("\n⏳ Waiting 5 seconds before next cycle...")
            time.sleep(5)
    
    # Show summary
    print("\n\n--- Monitoring Summary ---")
    summary = alert_mgr.get_alerts_summary(hours=24)
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Critical: {summary.get('critical_count', 0)}")
    print(f"Urgent: {summary.get('urgent_count', 0)}")
    print(f"Potential Cost Avoided: ${summary.get('total_prevented_cost', '$0')}")


def example_maintenance_report():
    """Example 4: Generate maintenance report"""
    print("\n\n" + "="*70)
    print("EXAMPLE 4: Maintenance Report Generation")
    print("="*70)
    
    API_ENDPOINT = "http://localhost:5001"
    
    try:
        response = requests.get(
            f"{API_ENDPOINT}/report/maintenance",
            timeout=10
        )
        report = response.json()
        
        print(f"\n📋 MAINTENANCE REPORT - {report['report_time']}")
        print("="*70)
        print(f"\nFleet Overview:")
        print(f"  Total arms monitored: {report['total_arms_monitored']}")
        print(f"  Arms with alerts: {report['arms_with_alerts']}")
        print(f"  Immediate action required: {report['immediate_action_required']}")
        print(f"  Urgent maintenance: {report['urgent_maintenance']}")
        print(f"  Average fleet health: {report['average_fleet_health']:.1f}%")
        
        print(f"\nCost-Benefit Analysis:")
        print(f"  Estimated downtime prevented: {report['estimated_downtime_avoided']}")
        print(f"  (Based on $5M loss per arm per day)")
        
        if report['critical_alerts']:
            print(f"\nTop Critical Alerts ({len(report['critical_alerts'])}):")
            print("-" * 70)
            for i, alert in enumerate(report['critical_alerts'][:5], 1):
                print(f"\n{i}. {alert['arm_id']}")
                print(f"   Failure Probability: {alert['failure_probability']:.1%}")
                print(f"   Priority: {alert['priority'].upper()}")
                print(f"   Health Score: {alert['health_score']:.1f}%")
                if 'failure_window' in alert:
                    win = alert['failure_window']
                    print(f"   Estimated Days: {win.get('estimated_days', 'N/A')}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        logger.error(f"Failed to get report: {str(e)}")


def example_alert_management():
    """Example 5: Alert management workflow"""
    print("\n\n" + "="*70)
    print("EXAMPLE 5: Alert Management & Notifications")
    print("="*70)
    
    alert_mgr = AlertManager()
    
    # Simulate high-risk prediction
    sample_assessment = {
        'failure_probability': 0.85,
        'risk_level': 'critical',
        'health_score': 15.0,
        'alert_needed': True,
        'maintenance_priority': 'immediate',
        'trend_analysis': {
            'warnings': [
                'Rapid temperature increase',
                'High vibration levels',
                'Pressure spike detected'
            ]
        },
        'predicted_failure_window': {
            'earliest': (datetime.now()).isoformat(),
            'latest': (datetime.now()).isoformat(),
            'estimated_days': 0.5,
            'confidence': 'high'
        }
    }
    
    # Create alert
    alert = alert_mgr.create_alert("ARM_490", sample_assessment)
    
    print("\n🚨 Alert Created:")
    print(f"  Arm ID: {alert['arm_id']}")
    print(f"  Severity: {alert['severity'].upper()}")
    print(f"  Failure Probability: {alert['failure_probability']:.1%}")
    print(f"  Alert ID: {alert['alert_id']}")
    
    print(f"\n📧 Notifications to Send:")
    
    # Simulate email notification
    success = alert_mgr.send_email_alert(
        alert,
        recipients=["maintenance@plant.com", "supervisor@plant.com"]
    )
    print(f"  Email: {'✅ Sent' if success else '❌ Failed'}")
    
    # Simulate Slack notification
    success = alert_mgr.send_slack_alert(
        alert,
        webhook_url="https://hooks.slack.com/services/..."
    )
    print(f"  Slack: {'✅ Posted' if success else '❌ Failed'}")
    
    # Get alerts summary
    print(f"\n📊 Alert Summary (Last 24 hours):")
    summary = alert_mgr.get_alerts_summary(hours=24)
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  Critical: {summary.get('critical_count', 0)}")
    print(f"  Urgent: {summary.get('urgent_count', 0)}")
    print(f"  Cost prevented: {summary['total_prevented_cost']}")


def example_real_world_scenario():
    """Example 6: Real-world scenario - bearing degradation over 24 hours"""
    print("\n\n" + "="*70)
    print("EXAMPLE 6: Real-World Scenario - Bearing Degradation")
    print("="*70 + "\n")
    
    API_ENDPOINT = "http://localhost:5001"
    
    # Simulate bearing degrading over 24-hour period
    hours_and_readings = [
        (0, {"temperature": 98.5, "vibration": 2.1, "pressure": 1013.0},
         "Normal operation - All systems green"),
        (4, {"temperature": 101.2, "vibration": 3.1, "pressure": 1025.0},
         "Slight increase - Still acceptable"),
        (12, {"temperature": 106.8, "vibration": 4.5, "pressure": 1080.0},
         "Escalating issues - Alert team to inspect"),
        (18, {"temperature": 112.5, "vibration": 6.2, "pressure": 1150.0},
         "Critical degradation - Schedule emergency maintenance"),
        (24, {"temperature": 118.0, "vibration": 7.8, "pressure": 1220.0},
         "Imminent failure - STOP OPERATION"),
    ]
    
    for hour, changes, situation in hours_and_readings:
        reading = {
            "arm_id": "ARM_250",
            "temperature": changes['temperature'],
            "pressure": changes['pressure'],
            "vibration": changes['vibration'],
            "humidity": 55.0,
            "power_consumption": 510.0 + (hour * 5)  # Power increases over time
        }
        
        print(f"\n⏰ Hour {hour}:")
        print(f"  Temperature: {reading['temperature']:.1f}°C")
        print(f"  Vibration: {reading['vibration']:.1f} mm/s")
        print(f"  Situation: {situation}")
        
        try:
            response = requests.post(
                f"{API_ENDPOINT}/predict/arm",
                json=reading,
                timeout=5
            )
            result = response.json()
            
            print(f"  Prediction:")
            print(f"    Failure Probability: {result['failure_probability']:.1%}")
            print(f"    Risk Level: {result['risk_level'].upper()}")
            print(f"    Health Score: {result['health_score']:.1f}%")
            
            if result['alert_needed']:
                print(f"    ⚠️ ACTION: {result['maintenance_priority'].upper()}")
        
        except Exception as e:
            print(f"  ❌ Prediction failed: {str(e)}")
    
    print("\n\n📊 Outcome:")
    print("✅ Bearing replaced during maintenance window")
    print("✅ Avoided unscheduled failure ($5M+ cost)")
    print("✅ Predictive system proved its value")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FACTORY GUARD AI - PRODUCTION EXAMPLES")
    print("500 Robotic Arms | 24-Hour Failure Prediction")
    print("="*70)
    
    print("\n🔗 Connecting to API at http://localhost:5001...")
    
    # Check API health
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy and ready\n")
        else:
            print("❌ API returned error")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        print("   Make sure to start it first: python api.py")
        return
    
    # Run examples
    try:
        example_single_arm_prediction()
        example_batch_prediction()
        example_monitoring_workflow()
        example_maintenance_report()
        example_alert_management()
        example_real_world_scenario()
        
        print("\n\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n📚 Next Steps:")
        print("1. Review QUICKSTART_PRODUCTION.md for deployment guide")
        print("2. Integrate with your sensor data source")
        print("3. Configure email/Slack alerts")
        print("4. Train maintenance team on alert response")
        print("5. Monitor ROI on prevented downtime")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {str(e)}")
        logger.error(f"Examples failed: {str(e)}")


if __name__ == "__main__":
    main()
