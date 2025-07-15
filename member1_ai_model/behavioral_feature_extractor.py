import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class BehavioralFeatures:
    """Data class for storing extracted behavioral features"""
    touch_patterns: np.ndarray
    motion_patterns: np.ndarray
    temporal_patterns: np.ndarray
    session_metadata: Dict
    timestamp: datetime

class TouchPatternExtractor:
    """Extract touch-based behavioral features"""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.touch_history = []
        
    def extract_touch_features(self, touch_events: List[Dict]) -> np.ndarray:
        """Extract touch patterns from touch events"""
        
        features = []
        
        for event in touch_events[-self.sequence_length:]:
            # Basic touch features
            touch_feature = [
                event.get('x', 0) / 1000.0,  # Normalized x coordinate
                event.get('y', 0) / 1000.0,  # Normalized y coordinate
                event.get('pressure', 0.5),  # Touch pressure
                event.get('size', 0.5),      # Touch size
                event.get('duration', 0.1),  # Touch duration
                self._calculate_velocity(event),  # Touch velocity
                self._calculate_acceleration(event),  # Touch acceleration
                event.get('action_type', 0)  # Action type (0: down, 1: move, 2: up)
            ]
            features.append(touch_feature)
        
        # Pad or truncate to sequence length
        features = self._pad_sequence(features, self.sequence_length, 8)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_velocity(self, event: Dict) -> float:
        """Calculate touch velocity"""
        if len(self.touch_history) < 2:
            return 0.0
        
        prev_event = self.touch_history[-1]
        dx = event.get('x', 0) - prev_event.get('x', 0)
        dy = event.get('y', 0) - prev_event.get('y', 0)
        dt = event.get('timestamp', 0) - prev_event.get('timestamp', 0)
        
        if dt > 0:
            velocity = math.sqrt(dx*dx + dy*dy) / dt
            return min(velocity / 1000.0, 1.0)  # Normalize
        return 0.0
    
    def _calculate_acceleration(self, event: Dict) -> float:
        """Calculate touch acceleration"""
        if len(self.touch_history) < 3:
            return 0.0
        
        # Calculate acceleration based on velocity changes
        current_vel = self._calculate_velocity(event)
        prev_vel = self._calculate_velocity(self.touch_history[-1])
        dt = event.get('timestamp', 0) - self.touch_history[-1].get('timestamp', 0)
        
        if dt > 0:
            acceleration = (current_vel - prev_vel) / dt
            return min(abs(acceleration) / 1000.0, 1.0)  # Normalize
        return 0.0
    
    def update_history(self, event: Dict):
        """Update touch history"""
        self.touch_history.append(event)
        if len(self.touch_history) > 100:  # Keep only recent history
            self.touch_history.pop(0)
    
    def _pad_sequence(self, sequence: List, target_length: int, feature_dim: int) -> List:
        """Pad or truncate sequence to target length"""
        if len(sequence) > target_length:
            return sequence[-target_length:]
        elif len(sequence) < target_length:
            padding = [[0.0] * feature_dim] * (target_length - len(sequence))
            return padding + sequence
        return sequence

class MotionPatternExtractor:
    """Extract motion-based behavioral features"""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.motion_history = []
        
    def extract_motion_features(self, sensor_data: List[Dict]) -> np.ndarray:
        """Extract motion patterns from sensor data"""
        
        features = []
        
        for data in sensor_data[-self.sequence_length:]:
            # Motion features from accelerometer and gyroscope
            motion_feature = [
                data.get('accel_x', 0) / 10.0,  # Normalized acceleration
                data.get('accel_y', 0) / 10.0,
                data.get('accel_z', 0) / 10.0,
                data.get('gyro_x', 0) / 5.0,   # Normalized rotation
                data.get('gyro_y', 0) / 5.0,
                data.get('gyro_z', 0) / 5.0,
            ]
            features.append(motion_feature)
        
        # Pad or truncate to sequence length
        features = self._pad_sequence(features, self.sequence_length, 6)
        
        return np.array(features, dtype=np.float32)
    
    def calculate_device_orientation(self, sensor_data: Dict) -> Dict:
        """Calculate device orientation metrics"""
        accel_x = sensor_data.get('accel_x', 0)
        accel_y = sensor_data.get('accel_y', 0)
        accel_z = sensor_data.get('accel_z', 0)
        
        # Calculate tilt angles
        pitch = math.atan2(-accel_x, math.sqrt(accel_y*accel_y + accel_z*accel_z))
        roll = math.atan2(accel_y, accel_z)
        
        # Calculate stability metrics
        magnitude = math.sqrt(accel_x*accel_x + accel_y*accel_y + accel_z*accel_z)
        stability = 1.0 - abs(magnitude - 9.81) / 9.81  # Stability relative to gravity
        
        return {
            'pitch': pitch,
            'roll': roll,
            'stability': max(0.0, min(1.0, stability))
        }
    
    def _pad_sequence(self, sequence: List, target_length: int, feature_dim: int) -> List:
        """Pad or truncate sequence to target length"""
        if len(sequence) > target_length:
            return sequence[-target_length:]
        elif len(sequence) < target_length:
            padding = [[0.0] * feature_dim] * (target_length - len(sequence))
            return padding + sequence
        return sequence

class TemporalPatternExtractor:
    """Extract temporal behavioral features"""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.session_start_time = datetime.now()
        self.action_history = []
        
    def extract_temporal_features(self, actions: List[Dict]) -> np.ndarray:
        """Extract temporal patterns from user actions"""
        
        features = []
        
        for action in actions[-self.sequence_length:]:
            # Temporal features
            temporal_feature = [
                self._get_time_of_day_feature(action.get('timestamp')),
                self._calculate_action_interval(action),
                self._calculate_session_progress(action),
                self._get_action_frequency(action.get('action_type'))
            ]
            features.append(temporal_feature)
        
        # Pad or truncate to sequence length
        features = self._pad_sequence(features, self.sequence_length, 4)
        
        return np.array(features, dtype=np.float32)
    
    def _get_time_of_day_feature(self, timestamp: Optional[float]) -> float:
        """Convert timestamp to time of day feature (0-1)"""
        if timestamp is None:
            return 0.5
        
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        # Convert to cyclical feature (0-1)
        return (hour % 24) / 24.0
    
    def _calculate_action_interval(self, action: Dict) -> float:
        """Calculate time interval between actions"""
        if not self.action_history:
            return 0.0
        
        current_time = action.get('timestamp', 0)
        previous_time = self.action_history[-1].get('timestamp', 0)
        
        interval = current_time - previous_time
        # Normalize to 0-1 range (assuming max 10 seconds between actions)
        return min(interval / 10.0, 1.0)
    
    def _calculate_session_progress(self, action: Dict) -> float:
        """Calculate progress within current session"""
        current_time = action.get('timestamp', 0)
        session_start = self.session_start_time.timestamp()
        
        progress = (current_time - session_start) / 300.0  # Normalize to 5 minutes
        return min(progress, 1.0)
    
    def _get_action_frequency(self, action_type: Optional[str]) -> float:
        """Get frequency of specific action type"""
        if not action_type or not self.action_history:
            return 0.0
        
        recent_actions = self.action_history[-20:]  # Last 20 actions
        count = sum(1 for a in recent_actions if a.get('action_type') == action_type)
        
        return count / len(recent_actions)
    
    def update_action_history(self, action: Dict):
        """Update action history"""
        self.action_history.append(action)
        if len(self.action_history) > 100:  # Keep only recent history
            self.action_history.pop(0)
    
    def _pad_sequence(self, sequence: List, target_length: int, feature_dim: int) -> List:
        """Pad or truncate sequence to target length"""
        if len(sequence) > target_length:
            return sequence[-target_length:]
        elif len(sequence) < target_length:
            padding = [[0.0] * feature_dim] * (target_length - len(sequence))
            return padding + sequence
        return sequence

class BehavioralFeatureExtractor:
    """Main class for extracting all behavioral features"""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.touch_extractor = TouchPatternExtractor(sequence_length)
        self.motion_extractor = MotionPatternExtractor(sequence_length)
        self.temporal_extractor = TemporalPatternExtractor(sequence_length)
        
    def extract_features(self, 
                        touch_events: List[Dict],
                        sensor_data: List[Dict],
                        user_actions: List[Dict],
                        session_metadata: Dict = None) -> BehavioralFeatures:
        """Extract all behavioral features from raw data"""
        
        # Extract individual feature types
        touch_patterns = self.touch_extractor.extract_touch_features(touch_events)
        motion_patterns = self.motion_extractor.extract_motion_features(sensor_data)
        temporal_patterns = self.temporal_extractor.extract_temporal_features(user_actions)
        
        # Update histories
        for event in touch_events:
            self.touch_extractor.update_history(event)
        
        for action in user_actions:
            self.temporal_extractor.update_action_history(action)
        
        # Create session metadata
        if session_metadata is None:
            session_metadata = {}
        
        session_metadata.update({
            'sequence_length': self.sequence_length,
            'extraction_timestamp': datetime.now().isoformat(),
            'touch_events_count': len(touch_events),
            'sensor_data_count': len(sensor_data),
            'user_actions_count': len(user_actions)
        })
        
        return BehavioralFeatures(
            touch_patterns=touch_patterns,
            motion_patterns=motion_patterns,
            temporal_patterns=temporal_patterns,
            session_metadata=session_metadata,
            timestamp=datetime.now()
        )
    
    def extract_features_for_inference(self, 
                                     touch_events: List[Dict],
                                     sensor_data: List[Dict],
                                     user_actions: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract features formatted for model inference"""
        
        features = self.extract_features(touch_events, sensor_data, user_actions)
        
        return {
            'touch_patterns': np.expand_dims(features.touch_patterns, axis=0),
            'motion_patterns': np.expand_dims(features.motion_patterns, axis=0),
            'temporal_patterns': np.expand_dims(features.temporal_patterns, axis=0)
        }
    
    def reset_extractors(self):
        """Reset all extractors for new session"""
        self.touch_extractor = TouchPatternExtractor(self.sequence_length)
        self.motion_extractor = MotionPatternExtractor(self.sequence_length)
        self.temporal_extractor = TemporalPatternExtractor(self.sequence_length)

def create_sample_data() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create sample data for testing"""
    
    # Sample touch events
    touch_events = []
    for i in range(60):
        touch_events.append({
            'x': 100 + i * 5,
            'y': 200 + i * 3,
            'pressure': 0.5 + (i % 10) * 0.05,
            'size': 0.3 + (i % 5) * 0.1,
            'duration': 0.1 + (i % 3) * 0.05,
            'action_type': i % 3,
            'timestamp': 1000000 + i * 100
        })
    
    # Sample sensor data
    sensor_data = []
    for i in range(60):
        sensor_data.append({
            'accel_x': 0.1 * math.sin(i * 0.1),
            'accel_y': 0.1 * math.cos(i * 0.1),
            'accel_z': 9.81 + 0.1 * math.sin(i * 0.2),
            'gyro_x': 0.01 * math.sin(i * 0.15),
            'gyro_y': 0.01 * math.cos(i * 0.15),
            'gyro_z': 0.005 * math.sin(i * 0.25),
            'timestamp': 1000000 + i * 100
        })
    
    # Sample user actions
    user_actions = []
    actions = ['tap', 'swipe', 'scroll', 'pinch']
    for i in range(60):
        user_actions.append({
            'action_type': actions[i % len(actions)],
            'timestamp': 1000000 + i * 100,
            'screen_area': f'area_{i % 5}',
            'duration': 0.2 + (i % 4) * 0.1
        })
    
    return touch_events, sensor_data, user_actions

if __name__ == "__main__":
    # Example usage
    extractor = BehavioralFeatureExtractor(sequence_length=50)
    
    # Create sample data
    touch_events, sensor_data, user_actions = create_sample_data()
    
    # Extract features
    features = extractor.extract_features(touch_events, sensor_data, user_actions)
    
    print("Feature extraction completed!")
    print(f"Touch patterns shape: {features.touch_patterns.shape}")
    print(f"Motion patterns shape: {features.motion_patterns.shape}")
    print(f"Temporal patterns shape: {features.temporal_patterns.shape}")
    
    # Extract features for inference
    inference_features = extractor.extract_features_for_inference(
        touch_events, sensor_data, user_actions
    )
    
    print("\nFeatures for inference:")
    for key, value in inference_features.items():
        print(f"{key}: {value.shape}")