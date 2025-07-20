"""
Create Sample AI Model for NETHRA Backend
This creates a dummy TensorFlow Lite model until the real one arrives from member 1
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_sample_trust_model():
    """Create a sample trust scoring model"""
    try:
        print("🤖 Creating sample AI model for NETHRA...")
        
        # Create a simple neural network for trust scoring
        model = tf.keras.Sequential([
            # Input: 50 behavioral features
            tf.keras.layers.Dense(64, activation='relu', input_shape=(50,), name='input_layer'),
            tf.keras.layers.Dropout(0.2),
            
            # Hidden layers for behavioral pattern analysis
            tf.keras.layers.Dense(32, activation='relu', name='behavioral_analysis'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(16, activation='relu', name='pattern_recognition'),
            tf.keras.layers.Dropout(0.1),
            
            # Output: Trust score (0-1, will be scaled to 0-100)
            tf.keras.layers.Dense(1, activation='sigmoid', name='trust_output')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Generate sample training data
        print("📊 Generating sample training data...")
        X_train = np.random.random((1000, 50))  # 1000 samples, 50 features
        
        # Create realistic trust scores based on feature patterns
        y_train = []
        for features in X_train:
            # Simulate trust scoring logic
            trust_score = 0.8  # Base trust
            
            # Reduce trust for unusual patterns
            if np.std(features) > 0.4:  # High variance = suspicious
                trust_score -= 0.3
            if np.mean(features) < 0.3:  # Low activity = suspicious
                trust_score -= 0.2
            if np.max(features) > 0.95:  # Perfect values = bot-like
                trust_score -= 0.4
                
            # Add some randomness
            trust_score += np.random.normal(0, 0.1)
            trust_score = max(0.0, min(1.0, trust_score))
            
            y_train.append(trust_score)
        
        y_train = np.array(y_train)
        
        # Train the model
        print("🎯 Training sample model...")
        model.fit(
            X_train, y_train,
            epochs=5,  # Quick training for demo
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Convert to TensorFlow Lite
        print("⚡ Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save the model
        model_path = Path(__file__).parent / "trust_model.tflite"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Sample model created: {model_path}")
        print(f"📏 Model size: {len(tflite_model) / 1024:.1f} KB")
        
        # Test the model
        print("🧪 Testing sample model...")
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        # Test with sample data
        test_input = np.random.random((1, 50)).astype(np.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        trust_score = float(output[0][0]) * 100
        
        print(f"🎯 Test prediction: {trust_score:.1f}% trust score")
        print("✅ Sample model is working perfectly!")
        
        return str(model_path)
        
    except Exception as e:
        logger.error(f"Failed to create sample model: {str(e)}")
        print(f"❌ Error creating model: {str(e)}")
        return None

if __name__ == "__main__":
    create_sample_trust_model()