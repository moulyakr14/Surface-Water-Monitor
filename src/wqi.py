import numpy as np
import pandas as pd
from src.config import WQI_WEIGHTS, WQI_LIMITS, WQI_CATEGORIES

def _quality_subindices(row):
    # Parameter-specific demo scoring, bounded to 0..200.
    ph_score = abs(float(row["ph"]) - 7.0) / WQI_LIMITS["ph_max_deviation"] * 100
    turb_score = float(row["turbidity_ntu"]) / WQI_LIMITS["turbidity_ntu"] * 100
    tds_score = float(row["tds_mg_l"]) / WQI_LIMITS["tds_mg_l"] * 100
    temp_score = max(0.0, float(row["temperature_c"]) - 20.0) / max(1.0, WQI_LIMITS["temperature_c"] - 20.0) * 100
    do_score = max(0.0, (WQI_LIMITS["dissolved_oxygen_mg_l"] - float(row["dissolved_oxygen_mg_l"])) /
                   WQI_LIMITS["dissolved_oxygen_mg_l"] * 100)
    return {
        "ph": np.clip(ph_score, 0, 200),
        "turbidity_ntu": np.clip(turb_score, 0, 200),
        "tds_mg_l": np.clip(tds_score, 0, 200),
        "temperature_c": np.clip(temp_score, 0, 200),
        "dissolved_oxygen_mg_l": np.clip(do_score, 0, 200),
    }

def category_for_wqi(wqi):
    for upper, label in WQI_CATEGORIES:
        if wqi <= upper:
            return label
    return "Unfit"

def calculate_wqi_row(row):
    sub = _quality_subindices(row)
    total_weight = sum(WQI_WEIGHTS.values())
    score = sum(sub[k] * WQI_WEIGHTS[k] for k in WQI_WEIGHTS) / total_weight
    return float(score), category_for_wqi(float(score))

def add_wqi_columns(df):
    required = list(WQI_WEIGHTS.keys())
    out = df.copy()
    scores, labels = [], []
    for _, row in out.iterrows():
        score, label = calculate_wqi_row(row)
        scores.append(score)
        labels.append(label)
    out["wqi"] = scores
    out["quality_category"] = labels
    return out
