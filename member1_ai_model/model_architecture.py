import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Dict, List

class BehavioralFeatureEncoder(layers.Layer):
    """Encodes behavioral features using CNN for sequential patterns"""
    
    def __init__(self, feature_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        
        # CNN layers for temporal pattern recognition
        self.conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')
        self.conv3 = layers.Conv1D(256, 3, activation='relu', padding='same')
        
        # Batch normalization for stability
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        
        # Dropout for regularization
        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.3)
        
        # Global pooling to reduce dimensions
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Dense layer for feature projection
        self.feature_projection = layers.Dense(feature_dim, activation='relu')
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        x = self.global_pool(x)
        x = self.feature_projection(x)
        
        return x

class TransformerEncoder(layers.Layer):
    """Transformer encoder for capturing long-range dependencies in behavior"""
    
    def __init__(self, d_model: int = 128, num_heads: int = 4, ff_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=0.1
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        
    def call(self, inputs, training=None, mask=None):
        # Multi-head attention
        attn_output = self.attention(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class GraphNeuralNetwork(layers.Layer):
    """Graph Neural Network for modeling user behavior relationships"""
    
    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.node_projection = layers.Dense(hidden_dim, activation='relu')
        self.edge_projection = layers.Dense(hidden_dim, activation='relu')
        
        # Message passing layers
        self.message_layer = layers.Dense(hidden_dim, activation='relu')
        self.update_layer = layers.Dense(hidden_dim, activation='relu')
        
        # Aggregation layer
        self.aggregation = layers.Dense(hidden_dim, activation='relu')
        
    def call(self, node_features, edge_features, adjacency_matrix, training=None):
        # Project node and edge features
        nodes = self.node_projection(node_features)
        edges = self.edge_projection(edge_features)
        
        # Message passing
        messages = self.message_layer(tf.concat([nodes, edges], axis=-1))
        
        # Aggregate messages using adjacency matrix
        aggregated = tf.matmul(adjacency_matrix, messages)
        
        # Update node representations
        updated_nodes = self.update_layer(nodes + aggregated)
        
        # Final aggregation
        graph_embedding = self.aggregation(tf.reduce_mean(updated_nodes, axis=1))
        
        return graph_embedding

class TrustProfileModel(Model):
    """Complete TrustProfile model using CNN-Transformer-GNN architecture"""
    
    def __init__(self, 
                 feature_dim: int = 128,
                 num_transformer_layers: int = 2,
                 num_heads: int = 4,
                 ff_dim: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Feature encoders
        self.touch_encoder = BehavioralFeatureEncoder(feature_dim, name='touch_encoder')
        self.motion_encoder = BehavioralFeatureEncoder(feature_dim, name='motion_encoder')
        self.temporal_encoder = BehavioralFeatureEncoder(feature_dim, name='temporal_encoder')
        
        # Transformer layers
        self.transformer_layers = [
            TransformerEncoder(feature_dim, num_heads, ff_dim, name=f'transformer_{i}')
            for i in range(num_transformer_layers)
        ]
        
        # Graph neural network
        self.gnn = GraphNeuralNetwork(feature_dim, name='gnn')
        
        # Trust score calculation layers
        self.trust_dense1 = layers.Dense(256, activation='relu', name='trust_dense1')
        self.trust_dense2 = layers.Dense(128, activation='relu', name='trust_dense2')
        self.trust_dense3 = layers.Dense(64, activation='relu', name='trust_dense3')
        
        # Output layers
        self.trust_score = layers.Dense(1, activation='sigmoid', name='trust_score')
        self.anomaly_score = layers.Dense(1, activation='sigmoid', name='anomaly_score')
        self.risk_category = layers.Dense(3, activation='softmax', name='risk_category')  # Low, Medium, High
        
        # Dropout for regularization
        self.dropout = layers.Dropout(0.2)
        
    def call(self, inputs, training=None):
        # Extract different behavioral features
        touch_features = inputs['touch_patterns']  # Shape: (batch, seq_len, touch_dim)
        motion_features = inputs['motion_patterns']  # Shape: (batch, seq_len, motion_dim)
        temporal_features = inputs['temporal_patterns']  # Shape: (batch, seq_len, temporal_dim)
        
        # Encode behavioral features
        touch_encoded = self.touch_encoder(touch_features, training=training)
        motion_encoded = self.motion_encoder(motion_features, training=training)
        temporal_encoded = self.temporal_encoder(temporal_features, training=training)
        
        # Combine features
        combined_features = tf.stack([touch_encoded, motion_encoded, temporal_encoded], axis=1)
        
        # Apply transformer layers
        transformer_output = combined_features
        for transformer_layer in self.transformer_layers:
            transformer_output = transformer_layer(transformer_output, training=training)
        
        # Graph neural network processing
        # Create simple adjacency matrix for behavioral features
        batch_size = tf.shape(combined_features)[0]
        adjacency_matrix = tf.ones((batch_size, 3, 3)) / 3.0  # Fully connected graph
        
        # Prepare node and edge features for GNN
        node_features = transformer_output
        edge_features = tf.tile(tf.expand_dims(tf.reduce_mean(transformer_output, axis=1), axis=1), [1, 3, 1])
        
        graph_embedding = self.gnn(node_features, edge_features, adjacency_matrix, training=training)
        
        # Trust score calculation
        x = self.trust_dense1(graph_embedding)
        x = self.dropout(x, training=training)
        x = self.trust_dense2(x)
        x = self.dropout(x, training=training)
        x = self.trust_dense3(x)
        
        # Generate outputs
        trust_score = self.trust_score(x) * 100  # Scale to 0-100
        anomaly_score = self.anomaly_score(x)
        risk_category = self.risk_category(x)
        
        return {
            'trust_score': trust_score,
            'anomaly_score': anomaly_score,
            'risk_category': risk_category,
            'embeddings': graph_embedding
        }

def create_trust_model(input_shapes: Dict[str, Tuple[int, int]]) -> TrustProfileModel:
    """Create and compile the TrustProfile model"""
    
    # Define inputs
    touch_input = layers.Input(shape=input_shapes['touch_patterns'], name='touch_patterns')
    motion_input = layers.Input(shape=input_shapes['motion_patterns'], name='motion_patterns')
    temporal_input = layers.Input(shape=input_shapes['temporal_patterns'], name='temporal_patterns')
    
    inputs = {
        'touch_patterns': touch_input,
        'motion_patterns': motion_input,
        'temporal_patterns': temporal_input
    }
    
    # Create model
    model = TrustProfileModel()
    
    # Build the model
    outputs = model(inputs)
    
    # Create functional model
    functional_model = Model(inputs=inputs, outputs=outputs, name='TrustProfileModel')
    
    # Compile model
    functional_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'trust_score': 'mse',
            'anomaly_score': 'binary_crossentropy',
            'risk_category': 'categorical_crossentropy'
        },
        metrics={
            'trust_score': ['mae'],
            'anomaly_score': ['accuracy'],
            'risk_category': ['accuracy']
        },
        loss_weights={
            'trust_score': 1.0,
            'anomaly_score': 0.5,
            'risk_category': 0.3
        }
    )
    
    return functional_model

def convert_to_tflite(model: Model, output_path: str = 'trust_model.tflite'):
    """Convert trained model to TensorFlow Lite format for mobile deployment"""
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings for mobile deployment
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Use float16 for better performance
    
    # Representative dataset for quantization (if needed)
    def representative_dataset():
        for _ in range(100):
            yield [
                np.random.random((1, 50, 8)).astype(np.float32),  # touch_patterns
                np.random.random((1, 50, 6)).astype(np.float32),  # motion_patterns
                np.random.random((1, 50, 4)).astype(np.float32),  # temporal_patterns
            ]
    
    converter.representative_dataset = representative_dataset
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted to TensorFlow Lite: {output_path}")
    return tflite_model

if __name__ == "__main__":
    # Example usage
    input_shapes = {
        'touch_patterns': (50, 8),    # 50 time steps, 8 touch features
        'motion_patterns': (50, 6),   # 50 time steps, 6 motion features
        'temporal_patterns': (50, 4)  # 50 time steps, 4 temporal features
    }
    
    model = create_trust_model(input_shapes)
    print("TrustProfile model created successfully!")
    print(model.summary())