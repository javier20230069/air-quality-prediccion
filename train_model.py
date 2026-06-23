from pathlib import Path
import io
import json
import zipfile
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
TARGET = "CO(GT)"
SELECTED_FEATURES = ["PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "NO2(GT)", "PT08.S5(O3)"]

FEATURE_LABELS = {
    "PT08.S1(CO)": "Sensor PT08.S1(CO)",
    "C6H6(GT)": "Concentracion de benceno C6H6(GT)",
    "PT08.S2(NMHC)": "Sensor PT08.S2(NMHC)",
    "NOx(GT)": "Oxidos de nitrogeno NOx(GT)",
    "NO2(GT)": "Dioxido de nitrogeno NO2(GT)",
    "PT08.S5(O3)": "Sensor PT08.S5(O3)",
}
FEATURE_HELP = {
    "PT08.S1(CO)": "Respuesta del sensor relacionado con monoxido de carbono.",
    "C6H6(GT)": "Concentracion de benceno registrada en la medicion.",
    "PT08.S2(NMHC)": "Respuesta del sensor relacionado con hidrocarburos no metanicos.",
    "NOx(GT)": "Concentracion de oxidos de nitrogeno.",
    "NO2(GT)": "Concentracion de dioxido de nitrogeno.",
    "PT08.S5(O3)": "Respuesta del sensor relacionado con ozono.",
}

def load_data():
    raw_data = urllib.request.urlopen(DATA_URL).read()
    zip_file = zipfile.ZipFile(io.BytesIO(raw_data))
    df = pd.read_csv(zip_file.open("AirQualityUCI.csv"), sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].replace(-200, np.nan)
    df = df.dropna(subset=[TARGET]).copy()
    return df

def build_model():
    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), SELECTED_FEATURES)],
        verbose_feature_names_out=False,
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1, max_depth=18, min_samples_leaf=1)),
    ])
    return model

def train_and_save(output_dir="."):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X = df[SELECTED_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {"mae": float(mean_absolute_error(y_test, predictions)), "r2": float(r2_score(y_test, predictions))}
    feature_metadata = []
    for feature in SELECTED_FEATURES:
        values = X[feature].dropna()
        feature_metadata.append({
            "name": feature,
            "label": FEATURE_LABELS[feature],
            "help": FEATURE_HELP[feature],
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        })
    metadata = {
        "dataset": "Air Quality Dataset",
        "source": DATA_URL,
        "target": TARGET,
        "target_description": "Concentracion real de monoxido de carbono en el aire.",
        "model": "RandomForestRegressor",
        "selected_features": SELECTED_FEATURES,
        "feature_metadata": feature_metadata,
        "metrics": metrics,
    }
    joblib.dump(model, output_path / "model.pkl")
    (output_path / "feature_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

if __name__ == "__main__":
    info = train_and_save()
    print("Modelo guardado en model.pkl")
    print("Variables seleccionadas:", ", ".join(info["selected_features"]))
    print("MAE:", round(info["metrics"]["mae"], 4))
    print("R2:", round(info["metrics"]["r2"], 4))