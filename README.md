# Surface Water Analysis & Real-Time Monitoring

A software-first prototype based on the project synopsis. It supports simulated readings now and is structured so ESP32 readings can be connected later.

## Features
- Simulated multi-parameter water-quality dataset
- Data validation, missing-value handling and rolling features
- Configurable weighted-arithmetic Water Quality Index (WQI)
- Random Forest classification and regression; optional XGBoost comparison
- Isolation Forest anomaly detection
- SQLite history database
- Streamlit dashboard with trends, alerts, CSV upload and manual reading entry

> Important: The built-in data is synthetic and is for software development/demo only. It is not field data and must not be used to make drinking-water or public-health decisions. WQI reference values and category boundaries are configurable demo assumptions. Confirm the standards, parameter limits, units and WQI method with your project guide and the relevant BIS/CPCB source before reporting results. A model trained on synthetic/rule-generated labels demonstrates the pipeline; it does not establish real-world predictive accuracy.

## 1. Setup
Install Python 3.10 or newer. Open this folder in VS Code, then run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Generate data and train models
```bash
python main.py
```
This creates `data/water_quality.csv`, preprocesses it, calculates WQI, stores readings in SQLite, and trains models into `models/`.

You can also run individual steps:
```bash
python -m src.generate_data
python -m src.pipeline
python -m src.train_models
```

## 3. Launch dashboard
```bash
streamlit run dashboard/app.py
```

## 4. Use the dashboard
- View the latest simulated/recorded values, WQI and category.
- Inspect trends and flagged readings.
- Upload a CSV using the same column names listed below.
- Add a manual reading for testing.
- Download the processed history.

Required input columns:
`timestamp, site, ph, turbidity_ntu, tds_mg_l, temperature_c, dissolved_oxygen_mg_l, water_level_m`

Optional `latitude` and `longitude` columns are supported.

## 5. Later ESP32 integration
The dashboard currently uses a local SQLite database. A future API can receive ESP32 JSON over HTTP/MQTT and call the same `save_reading()` function in `src/database.py`. Do not connect hardware until the software pipeline is working.

Example future payload:
```json
{
  "timestamp": "2026-11-12T12:00:00Z",
  "site": "Lake Site 1",
  "ph": 7.2,
  "turbidity_ntu": 4.1,
  "tds_mg_l": 260,
  "temperature_c": 27.0,
  "dissolved_oxygen_mg_l": 7.0,
  "water_level_m": 1.8
}
```

## WQI configuration
Edit `src/config.py`. The demo implementation uses parameter-specific quality ratings:
- pH: ideal 7.0; score increases with distance from ideal relative to configured allowable deviation.
- Turbidity and TDS: higher values increase the sub-index.
- Dissolved oxygen: lower values increase the sub-index.
- Temperature: higher values increase the demo sub-index relative to a configurable reference.
- Water level is monitored and anomaly-checked but is not included in this demo WQI because a universal water-quality rating for level is not defined here.

The weighted average is `WQI = sum(weight_i * subindex_i) / sum(weight_i)`.
This is an explicitly configurable project implementation, not a claim that one universal WQI method applies to all surface waters.
# Surface-Water-Monitor
