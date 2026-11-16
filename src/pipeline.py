import pandas as pd
from src.config import DATA_DIR
from src.generate_data import generate_dataset
from src.preprocessing import preprocess_data
from src.wqi import add_wqi_columns
from src.database import save_readings
from src.train_models import train

def run_pipeline():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "water_quality.csv"
    if not csv_path.exists():
        generate_dataset()
    raw = pd.read_csv(csv_path)
    clean = preprocess_data(raw)
    scored = add_wqi_columns(clean)
    scored["anomaly"] = 0
    scored.to_csv(DATA_DIR / "processed_data.csv", index=False)
    save_readings(scored)
    train()
    print("Pipeline complete. Processed data and models are ready.")

if __name__ == "__main__":
    run_pipeline()
