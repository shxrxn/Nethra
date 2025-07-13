import numpy as np
import tensorflow as tf
import joblib

# Load the TFLite model and scaler
interpreter = tf.lite.Interpreter(model_path="model/trust_model.tflite")
interpreter.allocate_tensors()

scaler = joblib.load("model/scaler.pkl")  # Make sure you have this

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_ai_model(input_data):
    raw_input = np.array([[input_data["swipe_speed"], input_data["tilt_angle"], input_data["hold_duration"]]])
    scaled_input = scaler.transform(raw_input).reshape(1, 3, 1).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], scaled_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return float(output_data[0][0])  # Anomaly score
