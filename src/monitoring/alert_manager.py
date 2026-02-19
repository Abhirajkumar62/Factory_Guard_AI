"""
Alerting System for Predictive Maintenance
Monitors predictions and triggers maintenance alerts.
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict
from enum import Enum

from src.utils.config import Logger, ConfigLoader


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"  # Failure within hours
    URGENT = "urgent"  # Failure within 24 hours
    HIGH = "high"  # Failure within 1 week
    MEDIUM = "medium"  # Routine maintenance


class AlertChannel(Enum):
    """Alert notification channels."""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DATABASE = "database"
    DASHBOARD = "dashboard"


class AlertManager:
    """Manages maintenance alerts and notifications."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize alert manager.

        Args:
            config_path: Path to configuration file
        """
        self.logger = Logger()
        self.config = ConfigLoader(config_path).config

        self.alerts_log = []
        self.notifications_sent = []

        self.logger.info("✅ Alert Manager initialized")

    def create_alert(self, arm_id: str, health_assessment: Dict) -> Dict:
        """
        Create maintenance alert from health assessment.

        Args:
            arm_id: Robotic arm identifier
            health_assessment: Health assessment from prediction engine

        Returns:
            Alert dictionary
        """
        failure_prob = health_assessment["failure_probability"]
        priority = health_assessment["maintenance_priority"]

        # Determine severity
        if failure_prob > 0.8:
            severity = AlertSeverity.CRITICAL
        elif failure_prob > 0.5:
            severity = AlertSeverity.URGENT
        elif failure_prob > 0.2:
            severity = AlertSeverity.HIGH
        else:
            severity = AlertSeverity.MEDIUM

        alert = {
            "alert_id": f"{arm_id}_{datetime.now().timestamp()}",
            "arm_id": arm_id,
            "severity": severity.value,
            "priority": priority,
            "failure_probability": failure_prob,
            "health_score": health_assessment["health_score"],
            "timestamp": datetime.now().isoformat(),
            "predicted_failure_window": health_assessment["predicted_failure_window"],
            "trend_warnings": health_assessment["trend_analysis"].get("warnings", []),
            "estimated_replacement_cost": self._estimate_cost(arm_id),
            "estimated_downtime_hours": self._estimate_downtime(),
            "maintenance_action": self._recommend_action(priority),
        }

        self.alerts_log.append(alert)
        return alert

    def _estimate_cost(self, arm_id: str) -> str:
        """Estimate cost of unplanned downtime."""
        # Manufacturing plant loses $5M per robotic arm per day
        daily_loss = 5_000_000
        estimated_hours = 24  # Assume 24-hour failure window
        estimated_cost = daily_loss * (estimated_hours / 24)
        return f"${estimated_cost:,.0f}"

    def _estimate_downtime(self) -> float:
        """Estimate hours of unplanned downtime if not maintained."""
        return 24.0  # Hours

    def _recommend_action(self, priority: str) -> str:
        """Recommend maintenance action."""
        actions = {
            "immediate": "STOP OPERATION - Schedule emergency maintenance within 4 hours",
            "urgent": "Alert maintenance team - Schedule maintenance within 12 hours",
            "high": "Create work order - Schedule maintenance within 24 hours",
            "routine": "Log for routine maintenance scheduling",
        }
        return actions.get(priority, "Monitor and reassess")

    def send_email_alert(self, alert: Dict, recipients: List[str]) -> bool:
        """
        Send email notification of maintenance alert.

        Args:
            alert: Alert dictionary
            recipients: Email addresses to notify

        Returns:
            Success status
        """
        try:
            # Get email config
            email_config = self.config.get("notifications.email", {})
            sender = email_config.get("sender")
            smtp_server = email_config.get("smtp_server")
            smtp_port = email_config.get("smtp_port")

            if not all([sender, smtp_server]):
                self.logger.warning("Email config not set, skipping email notification")
                return False

            # Compose email
            subject = (
                f"🚨 ALERT: {alert['arm_id']} - {alert['severity'].upper()} Priority"
            )

            body = f"""
PREDICTIVE MAINTENANCE ALERT
{'='*60}

Arm ID: {alert['arm_id']}
Severity: {alert['severity'].upper()}
Priority: {alert['priority'].upper()}
Timestamp: {alert['timestamp']}

Health Metrics:
- Health Score: {alert['health_score']:.1f}%
- Failure Probability: {alert['failure_probability']:.1%}
- Estimated Cost if Unattended: {alert['estimated_replacement_cost']}
- Predicted Failure Window: {alert['predicted_failure_window']['earliest']} to {alert['predicted_failure_window']['latest']}

Trend Warnings:
{chr(10).join(['  • ' + w for w in alert['trend_warnings']])}

RECOMMENDED ACTION:
{alert['maintenance_action']}

Alert ID: {alert['alert_id']}

---
Factory Guard AI Predictive Maintenance System
            """

            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # Send email (would connect to actual SMTP in production)
            self.logger.info(f"📧 Email alert would be sent to {recipients}")
            self.notifications_sent.append(
                {
                    "type": "email",
                    "recipients": recipients,
                    "alert_id": alert["alert_id"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def send_slack_alert(self, alert: Dict, webhook_url: str) -> bool:
        """
        Send Slack notification.

        Args:
            alert: Alert dictionary
            webhook_url: Slack webhook URL

        Returns:
            Success status
        """
        try:
            import requests

            # Color code by severity
            color_map = {
                "critical": "#FF0000",  # Red
                "urgent": "#FFA500",  # Orange
                "high": "#FFD700",  # Gold
                "medium": "#3498db",  # Blue
            }

            slack_payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert["severity"], "#cccccc"),
                        "title": f"🚨 Predictive Maintenance Alert: {alert['arm_id']}",
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert["severity"].upper(),
                                "short": True,
                            },
                            {
                                "title": "Priority",
                                "value": alert["priority"].upper(),
                                "short": True,
                            },
                            {
                                "title": "Failure Probability",
                                "value": f"{alert['failure_probability']:.1%}",
                                "short": True,
                            },
                            {
                                "title": "Health Score",
                                "value": f"{alert['health_score']:.1f}%",
                                "short": True,
                            },
                            {
                                "title": "Estimated Cost",
                                "value": alert["estimated_replacement_cost"],
                                "short": True,
                            },
                            {
                                "title": "Recommendation",
                                "value": alert["maintenance_action"],
                                "short": False,
                            },
                        ],
                        "footer": "Factory Guard AI",
                        "ts": int(datetime.now().timestamp()),
                    }
                ]
            }

            # In production, would POST to webhook_url
            self.logger.info(f"💬 Slack alert prepared for {alert['arm_id']}")
            self.notifications_sent.append(
                {
                    "type": "slack",
                    "alert_id": alert["alert_id"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

    def get_alerts_summary(self, hours: int = 24) -> Dict:
        """
        Get summary of alerts in past N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            a
            for a in self.alerts_log
            if datetime.fromisoformat(a["timestamp"]) > cutoff_time
        ]

        severity_counts = {}
        for alert in recent_alerts:
            severity = alert["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Calculate total potential downtime cost
        total_cost = sum(
            float(a["estimated_replacement_cost"].replace("$", "").replace(",", ""))
            for a in recent_alerts
        )

        return {
            "period_hours": hours,
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(recent_alerts),
            "by_severity": severity_counts,
            "critical_count": severity_counts.get("critical", 0),
            "urgent_count": severity_counts.get("urgent", 0),
            "total_prevented_cost": f"${total_cost:,.0f}",
            "affected_arms": len(set(a["arm_id"] for a in recent_alerts)),
            "recent_alerts": recent_alerts[-10:],  # Last 10
        }

    def export_alerts(self, filepath: str) -> None:
        """Export all alerts to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.alerts_log, f, indent=2)

        self.logger.info(f"✅ Exported {len(self.alerts_log)} alerts to {filepath}")
