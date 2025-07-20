"""
Behavioral Models - Data models for behavioral analysis
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum

class TrustLevel(str, Enum):
    """Trust level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class TouchPattern(BaseModel):
    """Touch pattern data model"""
    timestamp: float = Field(..., description="Timestamp of touch event")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    pressure: float = Field(..., ge=0, le=1, description="Touch pressure (0-1)")
    size: float = Field(..., ge=0, description="Touch size")
    duration: float = Field(..., ge=0, description="Touch duration in seconds")
    
    @validator('pressure', 'size', 'duration')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

class SwipePattern(BaseModel):
    """Swipe pattern data model"""
    timestamp: float = Field(..., description="Timestamp of swipe event")
    start_x: float = Field(..., description="Starting X coordinate")
    start_y: float = Field(..., description="Starting Y coordinate")
    end_x: float = Field(..., description="Ending X coordinate")
    end_y: float = Field(..., description="Ending Y coordinate")
    velocity: float = Field(..., ge=0, description="Swipe velocity")
    direction: float = Field(..., ge=0, le=360, description="Swipe direction in degrees")
    duration: float = Field(..., ge=0, description="Swipe duration in seconds")
    
    @validator('direction')
    def validate_direction(cls, v):
        if v < 0 or v > 360:
            raise ValueError('Direction must be between 0 and 360 degrees')
        return v

class DeviceMotion(BaseModel):
    """Device motion data model"""
    timestamp: float = Field(..., description="Timestamp of motion data")
    accelerometer_x: float = Field(..., description="X-axis acceleration")
    accelerometer_y: float = Field(..., description="Y-axis acceleration")
    accelerometer_z: float = Field(..., description="Z-axis acceleration")
    gyroscope_x: float = Field(..., description="X-axis rotation")
    gyroscope_y: float = Field(..., description="Y-axis rotation")
    gyroscope_z: float = Field(..., description="Z-axis rotation")
    magnetometer_x: Optional[float] = Field(None, description="X-axis magnetic field")
    magnetometer_y: Optional[float] = Field(None, description="Y-axis magnetic field")
    magnetometer_z: Optional[float] = Field(None, description="Z-axis magnetic field")

class AppUsage(BaseModel):
    """App usage data model"""
    session_duration: float = Field(..., ge=0, description="Session duration in seconds")
    screen_transitions: int = Field(..., ge=0, description="Number of screen transitions")
    button_clicks: int = Field(..., ge=0, description="Number of button clicks")
    scroll_events: int = Field(..., ge=0, description="Number of scroll events")
    text_input_events: int = Field(..., ge=0, description="Number of text input events")
    app_version: str = Field(..., description="App version")
    debug_mode: bool = Field(False, description="Debug mode enabled")
    root_detected: bool = Field(False, description="Root/jailbreak detected")
    hooking_detected: bool = Field(False, description="Hooking framework detected")
    memory_usage: float = Field(..., ge=0, description="Memory usage in MB")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")

class NetworkInfo(BaseModel):
    """Network information data model"""
    wifi_connected: bool = Field(..., description="WiFi connection status")
    network_type: str = Field(..., description="Network type (wifi, cellular, etc.)")
    signal_strength: float = Field(..., ge=0, le=1, description="Signal strength (0-1)")
    connection_speed: float = Field(..., ge=0, description="Connection speed in Mbps")
    ip_address: str = Field(..., description="IP address")
    proxy_detected: bool = Field(False, description="Proxy usage detected")
    vpn_detected: bool = Field(False, description="VPN usage detected")
    ssl_tampering_detected: bool = Field(False, description="SSL tampering detected")
    mitm_detected: bool = Field(False, description="MITM attack detected")
    dns_tampering_detected: bool = Field(False, description="DNS tampering detected")
    traffic_analysis_detected: bool = Field(False, description="Traffic analysis detected")
    localhost_detected: bool = Field(False, description="Localhost connection detected")

class LocationContext(BaseModel):
    """Location context data model"""
    is_home_location: bool = Field(False, description="Is user at home location")
    is_work_location: bool = Field(False, description="Is user at work location")
    location_confidence: float = Field(..., ge=0, le=1, description="Location confidence (0-1)")
    location_changed: bool = Field(False, description="Location changed recently")
    suspicious_location: bool = Field(False, description="Suspicious location detected")
    geofence_violation: bool = Field(False, description="Geofence violation detected")

class BehavioralFeatures(BaseModel):
    """Behavioral features extracted from raw data"""
    touch_speed_avg: float = Field(..., description="Average touch speed")
    touch_speed_std: float = Field(..., description="Touch speed standard deviation")
    touch_pressure_avg: float = Field(..., description="Average touch pressure")
    touch_pressure_std: float = Field(..., description="Touch pressure standard deviation")
    swipe_velocity_avg: float = Field(..., description="Average swipe velocity")
    swipe_velocity_std: float = Field(..., description="Swipe velocity standard deviation")
    device_tilt_avg: float = Field(..., description="Average device tilt")
    device_tilt_std: float = Field(..., description="Device tilt standard deviation")
    session_activity_level: float = Field(..., description="Session activity level")
    interaction_rhythm: float = Field(..., description="Interaction rhythm score")
    coordination_score: float = Field(..., description="Hand-eye coordination score")
    fatigue_indicator: float = Field(..., description="User fatigue indicator")

class TrustProfile(BaseModel):
    """User trust profile model"""
    user_id: str = Field(..., description="User identifier")
    created_at: datetime = Field(..., description="Profile creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    baseline_features: BehavioralFeatures = Field(..., description="Baseline behavioral features")
    confidence_level: float = Field(..., ge=0, le=1, description="Profile confidence level")
    adaptation_rate: float = Field(..., ge=0, le=1, description="Profile adaptation rate")
    anomaly_sensitivity: float = Field(..., ge=0, le=1, description="Anomaly detection sensitivity")
    update_count: int = Field(..., ge=0, description="Number of profile updates")

class AnomalyDetection(BaseModel):
    """Anomaly detection result model"""
    anomaly_type: str = Field(..., description="Type of anomaly detected")
    severity: RiskLevel = Field(..., description="Anomaly severity level")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    description: str = Field(..., description="Anomaly description")
    timestamp: datetime = Field(..., description="Detection timestamp")
    features_involved: List[str] = Field(..., description="Features involved in anomaly")
    recommendation: str = Field(..., description="Recommended action")

class TrustAssessment(BaseModel):
    """Trust assessment model"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(..., description="Assessment timestamp")
    trust_score: float = Field(..., ge=0, le=100, description="Trust score (0-100)")
    risk_level: RiskLevel = Field(..., description="Risk level")
    confidence: float = Field(..., ge=0, le=1, description="Assessment confidence")
    behavioral_features: BehavioralFeatures = Field(..., description="Behavioral features")
    anomalies: List[AnomalyDetection] = Field(..., description="Detected anomalies")
    contributing_factors: List[str] = Field(..., description="Trust score contributing factors")
    recommendations: List[str] = Field(..., description="Recommended actions")

class SessionData(BaseModel):
    """Session data model"""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    is_active: bool = Field(..., description="Session active status")
    trust_history: List[float] = Field(..., description="Trust score history")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    mirage_active: bool = Field(False, description="Mirage interface active")
    device_fingerprint: str = Field(..., description="Device fingerprint")
    location_context: Optional[LocationContext] = Field(None, description="Location context")

class MirageConfiguration(BaseModel):
    """Mirage interface configuration model"""
    mirage_type: str = Field(..., description="Type of mirage interface")
    intensity_level: float = Field(..., ge=0, le=1, description="Mirage intensity level")
    duration_minutes: int = Field(..., ge=1, le=120, description="Duration in minutes")
    fake_elements: List[Dict[str, Any]] = Field(..., description="Fake interface elements")
    cognitive_challenges: List[Dict[str, Any]] = Field(..., description="Cognitive challenges")
    adaptation_rules: Dict[str, Any] = Field(..., description="Adaptation rules")

class TamperDetectionResult(BaseModel):
    """Tamper detection result model"""
    tamper_detected: bool = Field(..., description="Tamper detection status")
    tamper_score: int = Field(..., ge=0, le=100, description="Tamper score (0-100)")
    severity: RiskLevel = Field(..., description="Tamper severity level")
    indicators: List[str] = Field(..., description="Tamper indicators")
    timestamp: datetime = Field(..., description="Detection timestamp")
    device_integrity: bool = Field(..., description="Device integrity status")
    app_integrity: bool = Field(..., description="App integrity status")
    network_integrity: bool = Field(..., description="Network integrity status")

class BehavioralAnalysisResult(BaseModel):
    """Behavioral analysis result model"""
    analysis_id: str = Field(..., description="Analysis identifier")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    trust_assessment: TrustAssessment = Field(..., description="Trust assessment")
    anomalies: List[AnomalyDetection] = Field(..., description="Detected anomalies")
    tamper_detection: TamperDetectionResult = Field(..., description="Tamper detection result")
    mirage_recommendation: Optional[MirageConfiguration] = Field(None, description="Mirage recommendation")
    privacy_compliance: bool = Field(..., description="Privacy compliance status")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class SystemMetrics(BaseModel):
    """System metrics model"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    active_sessions: int = Field(..., ge=0, description="Number of active sessions")
    total_analyses: int = Field(..., ge=0, description="Total analyses performed")
    average_trust_score: float = Field(..., ge=0, le=100, description="Average trust score")
    anomaly_detection_rate: float = Field(..., ge=0, le=1, description="Anomaly detection rate")
    false_positive_rate: float = Field(..., ge=0, le=1, description="False positive rate")
    mirage_activation_rate: float = Field(..., ge=0, le=1, description="Mirage activation rate")
    tamper_detection_count: int = Field(..., ge=0, description="Tamper detections today")
    system_uptime: float = Field(..., ge=0, description="System uptime in seconds")
    average_processing_time: float = Field(..., ge=0, description="Average processing time (ms)")
    battery_impact: float = Field(..., ge=0, le=100, description="Battery impact percentage")

class PrivacyMetrics(BaseModel):
    """Privacy metrics model"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    on_device_processing_rate: float = Field(..., ge=0, le=1, description="On-device processing rate")
    data_retention_compliance: bool = Field(..., description="Data retention compliance")
    encryption_coverage: float = Field(..., ge=0, le=1, description="Encryption coverage")
    anonymization_rate: float = Field(..., ge=0, le=1, description="Data anonymization rate")
    consent_compliance: bool = Field(..., description="Consent compliance status")
    privacy_score: float = Field(..., ge=0, le=100, description="Overall privacy score")
    gdpr_compliance: bool = Field(..., description="GDPR compliance status")
    dpdp_compliance: bool = Field(..., description="DPDP compliance status")

class DemoScenario(BaseModel):
    """Demo scenario model"""
    scenario_id: str = Field(..., description="Scenario identifier")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    attack_type: str = Field(..., description="Type of attack simulated")
    duration_seconds: int = Field(..., ge=1, description="Scenario duration")
    behavioral_patterns: List[Dict[str, Any]] = Field(..., description="Behavioral patterns")
    expected_trust_degradation: List[float] = Field(..., description="Expected trust degradation")
    mirage_activation_expected: bool = Field(..., description="Mirage activation expected")
    success_criteria: Dict[str, Any] = Field(..., description="Success criteria")