"""
AQI Hazard Alert System

Monitors AQI levels and triggers alerts when thresholds are exceeded.
Supports logging, webhook notifications, and in-app alert history.
"""
import json
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from src.config import AlertConfig


class AQIAlert:
    """Represents a single AQI alert event"""

    def __init__(
        self,
        aqi: float,
        category: str,
        level: str,
        message: str,
        location: str = "",
        timestamp: Optional[str] = None,
    ):
        self.aqi = aqi
        self.category = category
        self.level = level  # info | warning | alert | critical
        self.message = message
        self.location = location
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "aqi": self.aqi,
            "category": self.category,
            "level": self.level,
            "message": self.message,
            "location": self.location,
            "timestamp": self.timestamp,
        }


class AlertManager:
    """Manage AQI alerts: evaluate, log, notify"""

    def __init__(self):
        self.config = AlertConfig()
        self._last_alert_time: Dict[str, float] = {}  # level -> epoch
        self._alert_history: List[dict] = []
        self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, aqi: float, location: str = "Karachi") -> Optional[AQIAlert]:
        """
        Evaluate an AQI value and return an alert if warranted.

        Returns None when AQI is in the 'Good' range or alert is within
        cooldown window.
        """
        category, level = self._classify(aqi)

        # Only alert for warning/alert/critical levels
        if level == "info":
            return None

        # Cooldown check
        now = time.time()
        last = self._last_alert_time.get(level, 0)
        if now - last < self.config.cooldown_seconds:
            logger.debug(f"Alert suppressed (cooldown): {category} AQI={aqi}")
            return None

        message = self._build_message(aqi, category, level, location)
        alert = AQIAlert(
            aqi=aqi,
            category=category,
            level=level,
            message=message,
            location=location,
        )

        # Record & notify
        self._last_alert_time[level] = now
        self._alert_history.append(alert.to_dict())
        self._persist_alert(alert)
        self._notify(alert)

        logger.warning(f"AQI ALERT [{level.upper()}]: {message}")
        return alert

    def get_history(self, limit: int = 50) -> List[dict]:
        """Return recent alert history (newest first)."""
        return list(reversed(self._alert_history[-limit:]))

    def get_current_status(self, aqi: float) -> dict:
        """
        Return the status dict for a given AQI value (always, even
        in 'Good' range).  Useful for dashboard display.
        """
        category, level = self._classify(aqi)
        return {
            "aqi": aqi,
            "category": category,
            "level": level,
            "message": self._build_message(aqi, category, level),
            "recommendations": self._recommendations(category),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, aqi: float) -> tuple:
        """Return (category_name, level) for an AQI value."""
        for cat, info in self.config.thresholds.items():
            if info["min"] <= aqi <= info["max"]:
                return cat.replace("_", " ").title(), info["level"]
        return "Hazardous", "critical"

    @staticmethod
    def _build_message(aqi: float, category: str, level: str, location: str = "") -> str:
        loc = f" in {location}" if location else ""
        if level == "critical":
            return (
                f"HAZARDOUS AQI ({aqi:.0f}){loc}! "
                "Health emergency — everyone may experience serious effects. "
                "Avoid ALL outdoor activities."
            )
        if level == "alert":
            return (
                f"UNHEALTHY AQI ({aqi:.0f}){loc}. "
                "Everyone may begin to experience health effects. "
                "Sensitive groups should avoid outdoor exertion."
            )
        if level == "warning":
            return (
                f"AQI ({aqi:.0f}){loc} is Unhealthy for Sensitive Groups. "
                "Children, elderly, and people with respiratory conditions should limit outdoor exposure."
            )
        return f"AQI is {aqi:.0f}{loc} ({category})."

    @staticmethod
    def _recommendations(category: str) -> List[str]:
        recs = {
            "Good": ["Air quality is satisfactory. Enjoy outdoor activities."],
            "Moderate": [
                "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
            ],
            "Unhealthy Sensitive": [
                "Sensitive groups should reduce prolonged outdoor exertion.",
                "Keep windows closed.",
                "Consider using an air purifier indoors.",
            ],
            "Unhealthy": [
                "Everyone should reduce prolonged outdoor exertion.",
                "Wear an N95 mask if you must go outside.",
                "Keep windows and doors closed.",
                "Run air purifiers indoors.",
            ],
            "Very Unhealthy": [
                "Avoid ALL unnecessary outdoor activity.",
                "Wear an N95/P100 mask outdoors.",
                "Keep all windows sealed.",
                "Run HEPA air purifiers on high.",
                "Stay hydrated.",
            ],
            "Hazardous": [
                "HEALTH EMERGENCY — remain indoors.",
                "Seal windows and doors.",
                "Wear N95/P100 mask even indoors if air quality is poor.",
                "Contact healthcare provider if experiencing symptoms.",
                "Evacuate area if possible.",
            ],
        }
        return recs.get(category, recs["Hazardous"])

    def _persist_alert(self, alert: AQIAlert) -> None:
        """Append alert to the JSON log file."""
        try:
            log_path = self.config.alert_log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)

            history = []
            if log_path.exists():
                try:
                    history = json.loads(log_path.read_text())
                except (json.JSONDecodeError, Exception):
                    history = []

            history.append(alert.to_dict())
            # Keep last 500 alerts
            history = history[-500:]
            log_path.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")

    def _notify(self, alert: AQIAlert) -> None:
        """Send webhook notification if configured."""
        if not self.config.webhook_url:
            return
        try:
            payload = {
                "text": alert.message,
                "level": alert.level,
                **alert.to_dict(),
            }
            resp = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10,
            )
            logger.info(f"Webhook notification sent (status {resp.status_code})")
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")

    def _load_history(self) -> None:
        """Load persisted alert history from disk."""
        try:
            log_path = self.config.alert_log_path
            if log_path.exists():
                self._alert_history = json.loads(log_path.read_text())
        except Exception:
            self._alert_history = []
