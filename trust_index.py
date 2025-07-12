def calculate_trust_index(anomaly_score: float) -> int:
    trust_index = max(0, 100 - int(anomaly_score * 100))
    return trust_index