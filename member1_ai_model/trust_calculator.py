import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math

@dataclass
class TrustResult:
    """Result of trust calculation"""
    trust_score: float
    risk_level: str
    confidence: float
    anomaly_score: float
    contributing_factors: Dict[str, float]
    timestamp: datetime
    recommendations: List[str]

class TrustScoreCalculator:
    """Calculate trust scores and risk assessments"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.interpreter = None
        self.baseline_profile = None
        self.adaptation_rate = 0.1
        self.trust_threshold_high = 80.0
        self.trust_threshold_medium = 50.0
        self.trust_threshold_low = 20.0
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load TensorFlow Lite model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Model loaded successfully: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.interpreter = None
    
    def calculate_trust_score(self, features: Dict[str, np.ndarray]) -> TrustResult:
        """Calculate trust score from behavioral features"""
        
        if self.interpreter is None:
            # Fallback to heuristic calculation
            return self._calculate_heuristic_trust(features)
        
        try:
            # Prepare inputs for the model
            self._set_model_inputs(features)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get outputs
            outputs = self._get_model_outputs()
            
            # Process results
            return self._process_model_outputs(outputs, features)
            
        except Exception as e:
            print(f"Error in trust calculation: {e}")
            return self._calculate_heuristic_trust(features)
    
    def _set_model_inputs(self, features: Dict[str, np.ndarray]):
        """Set inputs for the TensorFlow Lite model"""
        
        input_mapping = {
            'touch_patterns': 0,
            'motion_patterns': 1,
            'temporal_patterns': 2
        }
        
        for feature_name, input_index in input_mapping.items():
            if feature_name in features:
                self.interpreter.set_tensor(
                    self.input_details[input_index]['index'],
                    features[feature_name].astype(np.float32)
                )
    
    def _get_model_outputs(self) -> Dict[str, np.ndarray]:
        """Get outputs from the TensorFlow Lite model"""
        
        outputs = {}
        output_names = ['trust_score', 'anomaly_score', 'risk_category']
        
        for i, name in enumerate(output_names):
            if i < len(self.output_details):
                outputs[name] = self.interpreter.get_tensor(
                    self.output_details[i]['index']
                )
        
        return outputs
    
    def _process_model_outputs(self, outputs: Dict[str, np.ndarray], 
                              features: Dict[str, np.ndarray]) -> TrustResult:
        """Process model outputs into trust result"""
        
        trust_score = float(outputs.get('trust_score', [50.0])[0])
        anomaly_score = float(outputs.get('anomaly_score', [0.0])[0])
        risk_category = outputs.get('risk_category', [[0.33, 0.33, 0.34]])[0]
        
        # Determine risk level
        risk_level = self._determine_risk_level(trust_score, risk_category)
        
        # Calculate confidence
        confidence = self._calculate_confidence(trust_score, anomaly_score)
        
        # Analyze contributing factors
        contributing_factors = self._analyze_contributing_factors(features, outputs)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trust_score, risk_level, contributing_factors)
        
        return TrustResult(
            trust_score=trust_score,
            risk_level=risk_level,
            confidence=confidence,
            anomaly_score=anomaly_score,
            contributing_factors=contributing_factors,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
    
    def _calculate_heuristic_trust(self, features: Dict[str, np.ndarray]) -> TrustResult:
        """Fallback heuristic trust calculation when model is not available"""
        
        trust_components = {}
        
        # Analyze touch patterns
        if 'touch_patterns' in features:
            trust_components['touch_consistency'] = self._analyze_touch_consistency(
                features['touch_patterns']
            )
        
        # Analyze motion patterns
        if 'motion_patterns' in features:
            trust_components['motion_stability'] = self._analyze_motion_stability(
                features['motion_patterns']
            )
        
        # Analyze temporal patterns
        if 'temporal_patterns' in features:
            trust_components['temporal_regularity'] = self._analyze_temporal_regularity(
                features['temporal_patterns']
            )
        
        # Calculate overall trust score
        trust_score = self._combine_trust_components(trust_components)
        
        # Determine risk level and other metrics
        risk_level = self._determine_risk_level(trust_score)
        confidence = min(0.8, max(0.4, trust_score / 100.0))  # Heuristic confidence
        anomaly_score = max(0.0, (100.0 - trust_score) / 100.0)
        
        recommendations = self._generate_recommendations(trust_score, risk_level, trust_components)
        
        return TrustResult(
            trust_score=trust_score,
            risk_level=risk_level,
            confidence=confidence,
            anomaly_score=anomaly_score,
            contributing_factors=trust_components,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
    
    def _analyze_touch_consistency(self, touch_patterns: np.ndarray) -> float:
        """Analyze consistency in touch patterns"""
        
        if touch_patterns.shape[0] == 0:
            return 50.0
        
        # Remove batch dimension if present
        if len(touch_patterns.shape) == 3:
            touch_patterns = touch_patterns[0]
        
        # Calculate variance in touch features
        touch_variance = np.var(touch_patterns, axis=0)
        
        # Pressure and size consistency (lower variance = higher trust)
        pressure_consistency = max(0.0, 1.0 - touch_variance[2] * 2.0)
        size_consistency = max(0.0, 1.0 - touch_variance[3] * 2.0)
        
        # Velocity consistency
        velocity_consistency = max(0.0, 1.0 - touch_variance[5] * 5.0)
        
        # Overall touch consistency
        consistency = (pressure_consistency + size_consistency + velocity_consistency) / 3.0
        
        return consistency * 100.0
    
    def _analyze_motion_stability(self, motion_patterns: np.ndarray) -> float:
        """Analyze stability in motion patterns"""
        
        if motion_patterns.shape[0] == 0:
            return 50.0
        
        # Remove batch dimension if present
        if len(motion_patterns.shape) == 3:
            motion_patterns = motion_patterns[0]
        
        # Calculate motion stability
        accel_variance = np.var(motion_patterns[:, :3], axis=0)
        gyro_variance = np.var(motion_patterns[:, 3:6], axis=0)
        
        # Lower variance in accelerometer indicates more stable holding
        accel_stability = max(0.0, 1.0 - np.mean(accel_variance) * 0.5)
        
        # Lower variance in gyroscope indicates less shaking
        gyro_stability = max(0.0, 1.0 - np.mean(gyro_variance) * 2.0)
        
        # Overall motion stability
        stability = (accel_stability + gyro_stability) / 2.0
        
        return stability * 100.0
    
    def _analyze_temporal_regularity(self, temporal_patterns: np.ndarray) -> float:
        """Analyze regularity in temporal patterns"""
        
        if temporal_patterns.shape[0] == 0:
            return 50.0
        
        # Remove batch dimension if present
        if len(temporal_patterns.shape) == 3:
            temporal_patterns = temporal_patterns[0]
        
        # Analyze action intervals (column 1)
        intervals = temporal_patterns[:, 1]
        interval_variance = np.var(intervals)
        
        # Regular intervals indicate normal behavior
        regularity = max(0.0, 1.0 - interval_variance * 2.0)
        
        # Analyze session progress (column 2)
        session_progress = temporal_patterns[:, 2]
        if len(session_progress) > 1:
            progress_consistency = 1.0 - abs(np.diff(session_progress).mean() - 0.02)
        else:
            progress_consistency = 0.5
        
        # Overall temporal regularity
        temporal_score = (regularity + max(0.0, progress_consistency)) / 2.0
        
        return temporal_score * 100.0
    
    def _combine_trust_components(self, components: Dict[str, float]) -> float:
        """Combine individual trust components into overall score"""
        
        if not components:
            return 50.0
        
        # Weighted combination of components
        weights = {
            'touch_consistency': 0.4,
            'motion_stability': 0.3,
            'temporal_regularity': 0.3
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, score in components.items():
            weight = weights.get(component, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        
        return 50.0
    
    def _determine_risk_level(self, trust_score: float, 
                            risk_category: Optional[np.ndarray] = None) -> str:
        """Determine risk level based on trust score"""
        
        if risk_category is not None:
            # Use model's risk category prediction
            risk_idx = np.argmax(risk_category)
            risk_levels = ['Low', 'Medium', 'High']
            return risk_levels[risk_idx]
        
        # Use trust score thresholds
        if trust_score >= self.trust_threshold_high:
            return 'Low'
        elif trust_score >= self.trust_threshold_medium:
            return 'Medium'
        else:
            return 'High'
    
    def _calculate_confidence(self, trust_score: float, anomaly_score: float) -> float:
        """Calculate confidence in the trust assessment"""
        
        # Confidence is higher when trust score is extreme (very high or very low)
        # and anomaly score is consistent with trust score
        
        score_extremity = abs(trust_score - 50.0) / 50.0
        anomaly_consistency = 1.0 - abs(anomaly_score - (100.0 - trust_score) / 100.0)
        
        confidence = (score_extremity + anomaly_consistency) / 2.0
        
        return max(0.1, min(1.0, confidence))
    
    def _analyze_contributing_factors(self, features: Dict[str, np.ndarray], 
                                    outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze factors contributing to trust score"""
        
        factors = {}
        
        # Analyze each feature type
        if 'touch_patterns' in features:
            factors['touch_behavior'] = self._analyze_touch_consistency(features['touch_patterns'])
        
        if 'motion_patterns' in features:
            factors['device_handling'] = self._analyze_motion_stability(features['motion_patterns'])
        
        if 'temporal_patterns' in features:
            factors['interaction_timing'] = self._analyze_temporal_regularity(features['temporal_patterns'])
        
        # Add model-specific factors if available
        if 'anomaly_score' in outputs:
            factors['anomaly_detection'] = (1.0 - float(outputs['anomaly_score'][0])) * 100.0
        
        return factors
    
    def _generate_recommendations(self, trust_score: float, risk_level: str, 
                                contributing_factors: Dict[str, float]) -> List[str]:
        """Generate recommendations based on trust assessment"""
        
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append("Consider additional authentication")
            recommendations.append("Monitor session closely")
            recommendations.append("Enable transaction limits")
        
        elif risk_level == 'Medium':
            recommendations.append("Verify recent transactions")
            recommendations.append("Check device security settings")
        
        else:  # Low risk
            recommendations.append("Continue normal operation")
            recommendations.append("Session appears secure")
        
        # Add specific recommendations based on contributing factors
        for factor, score in contributing_factors.items():
            if score < 40.0:
                if factor == 'touch_behavior':
                    recommendations.append("Unusual touch patterns detected")
                elif factor == 'device_handling':
                    recommendations.append("Device motion patterns seem irregular")
                elif factor == 'interaction_timing':
                    recommendations.append("Interaction timing appears unusual")
        
        return recommendations
    
    def update_baseline_profile(self, trust_result: TrustResult):
        """Update baseline profile based on trusted sessions"""
        
        if trust_result.trust_score >= self.trust_threshold_high:
            # Update baseline only with high-trust sessions
            if self.baseline_profile is None:
                self.baseline_profile = trust_result.contributing_factors.copy()
            else:
                # Exponential moving average update
                for factor, score in trust_result.contributing_factors.items():
                    if factor in self.baseline_profile:
                        self.baseline_profile[factor] = (
                            (1 - self.adaptation_rate) * self.baseline_profile[factor] +
                            self.adaptation_rate * score
                        )
                    else:
                        self.baseline_profile[factor] = score
    
    def get_trust_history_summary(self, trust_history: List[TrustResult]) -> Dict:
        """Generate summary of trust history"""
        
        if not trust_history:
            return {}
        
        trust_scores = [result.trust_score for result in trust_history]
        
        return {
            'average_trust': np.mean(trust_scores),
            'min_trust': np.min(trust_scores),
            'max_trust': np.max(trust_scores),
            'trust_trend': self._calculate_trust_trend(trust_scores),
            'risk_distribution': self._calculate_risk_distribution(trust_history),
            'session_count': len(trust_history)
        }
    
    def _calculate_trust_trend(self, trust_scores: List[float]) -> str:
        """Calculate trend in trust scores"""
        
        if len(trust_scores) < 2:
            return 'stable'
        
        # Calculate trend using linear regression
        x = np.arange(len(trust_scores))
        slope = np.polyfit(x, trust_scores, 1)[0]
        
        if slope > 5.0:
            return 'improving'
        elif slope < -5.0:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_risk_distribution(self, trust_history: List[TrustResult]) -> Dict[str, float]:
        """Calculate distribution of risk levels"""
        
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        
        for result in trust_history:
            risk_counts[result.risk_level] += 1
        
        total = len(trust_history)
        return {risk: count / total for risk, count in risk_counts.items()}

# Example usage and testing
if __name__ == "__main__":
    # Create trust calculator
    calculator = TrustScoreCalculator()
    
    # Create sample features
    sample_features = {
        'touch_patterns': np.random.random((1, 50, 8)),
        'motion_patterns': np.random.random((1, 50, 6)),
        'temporal_patterns': np.random.random((1, 50, 4))
    }
    
    # Calculate trust score
    trust_result = calculator.calculate_trust_score(sample_features)
    
    print("Trust Calculation Results:")
    print(f"Trust Score: {trust_result.trust_score:.2f}")
    print(f"Risk Level: {trust_result.risk_level}")
    print(f"Confidence: {trust_result.confidence:.2f}")
    print(f"Anomaly Score: {trust_result.anomaly_score:.2f}")
    print(f"Contributing Factors: {trust_result.contributing_factors}")
    print(f"Recommendations: {trust_result.recommendations}")