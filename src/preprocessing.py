import pandas as pd
from src.config import FEATURES

REQUIRED_COLUMNS = ["timestamp", "site"] + FEATURES

def preprocess_data(df):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing_cols))
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"])
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # Remove impossible readings before imputing; thresholds are broad validation guards.
    bounds = {
        "ph": (0, 14), "turbidity_ntu": (0, 1000), "tds_mg_l": (0, 10000),
        "temperature_c": (-5, 60), "dissolved_oxygen_mg_l": (0, 30),
        "water_level_m": (0, 100)
    }
    for col, (low, high) in bounds.items():
        out.loc[~out[col].between(low, high), col] = pd.NA
    out = out.sort_values("timestamp")
    out[FEATURES] = out[FEATURES].interpolate(limit_direction="both")
    out[FEATURES] = out[FEATURES].fillna(out[FEATURES].median(numeric_only=True))
    out["site"] = out["site"].fillna("Unknown").astype(str)
    out["turbidity_change"] = out.groupby("site")["turbidity_ntu"].diff().fillna(0)
    out["ph_deviation"] = (out["ph"] - 7.0).abs()
    out["rolling_turbidity_mean"] = out.groupby("site")["turbidity_ntu"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    return out.reset_index(drop=True)
