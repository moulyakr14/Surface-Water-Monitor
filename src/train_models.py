import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from src.config import FEATURES, MODEL_DIR, DATA_DIR
from src.generate_data import generate_dataset
from src.preprocessing import preprocess_data
from src.wqi import add_wqi_columns

def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "water_quality.csv"
    if not csv_path.exists():
        generate_dataset()
    raw = pd.read_csv(csv_path)
    df = add_wqi_columns(preprocess_data(raw))
    X = df[FEATURES]
    y = df["quality_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight="balanced",
                                 random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    metrics = {
        "classification_accuracy": float(accuracy_score(y_test, pred)),
        "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }
    joblib.dump(clf, MODEL_DIR / "quality_classifier.joblib")

    reg = RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, df["wqi"], test_size=0.2, random_state=42
    )
    reg.fit(Xr_train, yr_train)
    rp = reg.predict(Xr_test)
    metrics.update({
        "wqi_mae": float(mean_absolute_error(yr_test, rp)),
        "wqi_rmse": float(np.sqrt(mean_squared_error(yr_test, rp))),
        "wqi_r2": float(r2_score(yr_test, rp)),
    })
    joblib.dump(reg, MODEL_DIR / "wqi_regressor.joblib")

    anomaly = IsolationForest(contamination=0.04, n_estimators=200, random_state=42)
    anomaly.fit(X)
    joblib.dump(anomaly, MODEL_DIR / "anomaly_detector.joblib")

    # Optional XGBoost comparison; Random Forest remains available if XGBoost is absent.
    try:
        from xgboost import XGBClassifier
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y)
        Xa_train, Xa_test, ya_train, ya_test = train_test_split(
            X, encoded, test_size=0.2, random_state=42, stratify=encoded
        )
        xgb = XGBClassifier(n_estimators=180, max_depth=5, learning_rate=0.06,
                            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
                            random_state=42)
        xgb.fit(Xa_train, ya_train)
        xgb_pred = xgb.predict(Xa_test)
        metrics["xgboost_accuracy"] = float(accuracy_score(ya_test, xgb_pred))
        joblib.dump(xgb, MODEL_DIR / "xgboost_classifier.joblib")
        joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    except Exception as exc:
        metrics["xgboost_note"] = f"XGBoost comparison skipped: {exc}"

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Feature importance for a simple explainability view.
    importance = pd.DataFrame({"feature": FEATURES, "importance": clf.feature_importances_})
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    train()
