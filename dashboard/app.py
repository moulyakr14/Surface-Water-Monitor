import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import FEATURES, DATA_DIR
from src.generate_data import generate_dataset
from src.preprocessing import preprocess_data
from src.wqi import add_wqi_columns
from src.database import init_db, save_readings, load_readings
from src.predict import predict_reading

st.set_page_config(page_title="Surface Water Monitor", page_icon="💧", layout="wide")
st.title("💧 Surface Water Analysis & Monitoring")
st.caption("Software prototype • simulated readings until ESP32 integration")

init_db()
if not (DATA_DIR / "water_quality.csv").exists():
    generate_dataset()

with st.sidebar:
    st.header("Data controls")
    if st.button("Generate fresh demo dataset", use_container_width=True):
        generate_dataset()
        st.success("New synthetic dataset created. Run the pipeline again to retrain.")
    uploaded = st.file_uploader("Upload water-quality CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            clean = add_wqi_columns(preprocess_data(incoming))
            clean["anomaly"] = 0
            save_readings(clean)
            st.success(f"Imported {len(clean)} readings.")
        except Exception as e:
            st.error(f"Could not import CSV: {e}")

history = load_readings()
if history.empty:
    source = pd.read_csv(DATA_DIR / "water_quality.csv")
    clean = add_wqi_columns(preprocess_data(source))
    clean["anomaly"] = 0
    save_readings(clean)
    history = load_readings()

history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
latest = history.iloc[-1]

st.subheader("Latest water-quality status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("WQI score", f"{latest['wqi']:.1f}")
c2.metric("Quality category", str(latest["quality_category"]))
c3.metric("pH", f"{latest['ph']:.2f}")
c4.metric("Dissolved oxygen", f"{latest['dissolved_oxygen_mg_l']:.2f} mg/L")

st.subheader("Latest parameter readings")
param_cols = st.columns(6)
labels = [
    ("Turbidity", "turbidity_ntu", " NTU"),
    ("TDS", "tds_mg_l", " mg/L"),
    ("Temperature", "temperature_c", " °C"),
    ("Water level", "water_level_m", " m"),
    ("pH", "ph", ""),
    ("Dissolved oxygen", "dissolved_oxygen_mg_l", " mg/L")
]
for col, (label, key, unit) in zip(param_cols, labels):
    col.metric(label, f"{latest[key]:.2f}{unit}")

st.subheader("Water-quality trends")
chart_metric = st.selectbox("Choose parameter", ["wqi"] + FEATURES)
fig = px.line(history.tail(500), x="timestamp", y=chart_metric, color="site",
              title=f"{chart_metric.replace('_', ' ').title()} over time")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Quality category distribution")
    cat_counts = history["quality_category"].value_counts().rename_axis("category").reset_index(name="count")
    st.plotly_chart(px.pie(cat_counts, names="category", values="count", hole=.45),
                    use_container_width=True)
with right:
    st.subheader("Recent records")
    st.dataframe(history.tail(15).sort_values("timestamp", ascending=False),
                 use_container_width=True, hide_index=True)

st.subheader("Test a new reading")
with st.form("reading_form"):
    a, b, c = st.columns(3)
    ph = a.number_input("pH", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
    turbidity = b.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.5)
    tds = c.number_input("TDS (mg/L)", min_value=0.0, value=280.0, step=10.0)
    d, e, f = st.columns(3)
    temp = d.number_input("Temperature (°C)", min_value=-5.0, max_value=60.0, value=27.0, step=0.5)
    do = e.number_input("Dissolved oxygen (mg/L)", min_value=0.0, max_value=30.0, value=7.0, step=0.1)
    level = f.number_input("Water level (m)", min_value=0.0, value=1.8, step=0.1)
    site = st.text_input("Monitoring site", "Demo Lake")
    submitted = st.form_submit_button("Analyse reading")
if submitted:
    reading = {
        "ph": ph, "turbidity_ntu": turbidity, "tds_mg_l": tds,
        "temperature_c": temp, "dissolved_oxygen_mg_l": do, "water_level_m": level
    }
    result = predict_reading(reading)
    new_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "site": site,
        **reading, "wqi": result["wqi"], "quality_category": result["wqi_category"],
        "anomaly": int(result["anomaly"])
    }])
    save_readings(new_row)
    st.success(f"WQI: {result['wqi']:.1f} — {result['wqi_category']}")
    if "ml_category" in result:
        st.info(f"ML-predicted category: {result['ml_category']} (confidence {result.get('confidence', 0):.1%})")
    if result["anomaly"]:
        st.warning("Isolation Forest flagged this reading as unusual. Check whether it is pollution or a sensor/data issue.")
    st.rerun()

st.subheader("Alerts and data quality")
alerts = history[
    (history["quality_category"].isin(["Very Poor", "Unfit"])) |
    (history["anomaly"].fillna(0).astype(int) == 1)
]
if alerts.empty:
    st.success("No stored severe-category or anomaly flags in the current history.")
else:
    st.warning(f"{len(alerts)} stored record(s) need attention.")
    st.dataframe(alerts.tail(30), use_container_width=True, hide_index=True)

st.download_button(
    "Download stored readings as CSV",
    data=history.to_csv(index=False).encode("utf-8"),
    file_name="surface_water_history.csv",
    mime="text/csv"
)

st.caption("Demo WQI categories and synthetic labels are configurable examples, not certified water-safety assessments.")
