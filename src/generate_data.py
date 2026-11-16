import numpy as np
import pandas as pd
from src.config import DATA_DIR, FEATURES

def generate_dataset(rows=2500, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=rows, freq="15min")
    # Mix of normal, moderately impacted and polluted conditions for demo purposes.
    regime = rng.choice(["normal", "moderate", "polluted"], size=rows, p=[0.60, 0.27, 0.13])
    values = []
    for r in regime:
        if r == "normal":
            vals = [rng.normal(7.2, .25), rng.gamma(2, 1.2), rng.normal(280, 65),
                    rng.normal(26, 2.5), rng.normal(7.4, .9), rng.normal(1.8, .18)]
        elif r == "moderate":
            vals = [rng.normal(7.7, .55), rng.normal(8, 2.5), rng.normal(520, 110),
                    rng.normal(29, 3), rng.normal(5.2, 1), rng.normal(1.55, .25)]
        else:
            vals = [rng.normal(8.7, .6), rng.normal(18, 6), rng.normal(850, 180),
                    rng.normal(32, 3), rng.normal(3.2, .9), rng.normal(1.25, .3)]
        values.append(vals)
    df = pd.DataFrame(values, columns=FEATURES)
    df["ph"] = df["ph"].clip(4, 10)
    df["turbidity_ntu"] = df["turbidity_ntu"].clip(0, 100)
    df["tds_mg_l"] = df["tds_mg_l"].clip(10, 2500)
    df["temperature_c"] = df["temperature_c"].clip(5, 45)
    df["dissolved_oxygen_mg_l"] = df["dissolved_oxygen_mg_l"].clip(.1, 15)
    df["water_level_m"] = df["water_level_m"].clip(.05, 5)
    df.insert(0, "timestamp", timestamps.astype(str))
    df.insert(1, "site", "Demo Lake")
    # Small realistic data-quality issues to test preprocessing.
    for col in ["ph", "turbidity_ntu", "tds_mg_l"]:
        missing_idx = rng.choice(df.index, size=max(1, rows // 100), replace=False)
        df.loc[missing_idx, col] = np.nan
    out = DATA_DIR / "water_quality.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} synthetic readings: {out}")
    return out

if __name__ == "__main__":
    generate_dataset()
