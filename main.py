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
    label: int  # ✅ NEW PARAMETER

@app.get("/")
def read_root():
    return {"message": "NETHRA Backend is running. Use /docs to test the API."}

@app.post("/trust")
def evaluate_trust(data: InputData):
    input_dict = data.dict()
    anomaly_score = run_ai_model(input_dict)
    trust_index = calculate_trust_index(anomaly_score)
    tampering_detected = check_tampering(trust_index)

    return {
        "TrustIndex": trust_index,
        "TamperingDetected": tampering_detected,
        "AnomalyScore": round(anomaly_score, 3),
        "Label": input_dict["label"]
    }
