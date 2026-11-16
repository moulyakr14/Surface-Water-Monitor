import joblib
import pandas as pd
from src.config import FEATURES, MODEL_DIR
from src.wqi import calculate_wqi_row

def predict_reading(reading):
    row = pd.DataFrame([{k: float(reading[k]) for k in FEATURES}])
    result = {}
    # The formula-based WQI is the primary transparent score.
    score, category = calculate_wqi_row(row.iloc[0])
    result["wqi"] = score
    result["wqi_category"] = category
    model_path = MODEL_DIR / "quality_classifier.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        result["ml_category"] = str(model.predict(row[FEATURES])[0])
        if hasattr(model, "predict_proba"):
            result["confidence"] = float(max(model.predict_proba(row[FEATURES])[0]))
    anomaly_path = MODEL_DIR / "anomaly_detector.joblib"
    if anomaly_path.exists():
        detector = joblib.load(anomaly_path)
        result["anomaly"] = bool(detector.predict(row[FEATURES])[0] == -1)
    else:
        result["anomaly"] = False
    return result
