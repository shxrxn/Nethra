from fastapi import FastAPI
from pydantic import BaseModel
from ai_interface import run_ai_model
from trust_index import calculate_trust_index
from tamper_detection import check_tampering

app = FastAPI()

class InputData(BaseModel):
    swipe_speed: float
    tilt_angle: float
    hold_duration: float

@app.post("/trust")
def evaluate_trust(data: InputData):
    input_dict = data.dict()
    
    # Run TFLite model (from Member 1)
    anomaly_score = run_ai_model(input_dict)
    
    # Compute TrustIndex
    trust_index = calculate_trust_index(anomaly_score)
    
    # Check if tampering is suspected
    tampering_detected = check_tampering(trust_index)
    
    # Return results
    return {
        "TrustIndex": trust_index,
        "TamperingDetected": tampering_detected,
        "AnomalyScore": round(anomaly_score, 3)
    }

