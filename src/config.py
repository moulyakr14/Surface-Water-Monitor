from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "data" / "water_quality.db"

FEATURES = [
    "ph", "turbidity_ntu", "tds_mg_l", "temperature_c",
    "dissolved_oxygen_mg_l", "water_level_m"
]

# DEMO REFERENCE VALUES ONLY. Confirm the correct water-body/use-class standards
# with your faculty and authoritative BIS/CPCB documentation before final use.
WQI_WEIGHTS = {
    "ph": 0.20,
    "turbidity_ntu": 0.20,
    "tds_mg_l": 0.20,
    "temperature_c": 0.10,
    "dissolved_oxygen_mg_l": 0.30,
}
WQI_LIMITS = {
    "ph_max_deviation": 1.5,       # distance from ideal pH 7.0
    "turbidity_ntu": 5.0,         # demo reference value
    "tds_mg_l": 500.0,            # demo reference value
    "temperature_c": 30.0,        # demo reference value
    "dissolved_oxygen_mg_l": 5.0, # minimum reference value
}
# Demo categories; calibrate to the exact WQI convention adopted in the report.
WQI_CATEGORIES = [
    (25, "Excellent"),
    (50, "Good"),
    (75, "Poor"),
    (100, "Very Poor"),
    (float("inf"), "Unfit"),
]
