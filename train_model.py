from pathlib import Path
import io
import json
import zipfile
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
TARGET = "CO(GT)"

NUMERIC_FEATURES = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "hour",
    "month",
    "day_of_week",
]
CATEGORICAL_FEATURES = ["periodo_dia"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "PT08.S1(CO)": "Sensor PT08.S1(CO)",
    "C6H6(GT)": "Concentracion de benceno C6H6(GT)",
    "PT08.S2(NMHC)": "Sensor PT08.S2(NMHC)",
    "NOx(GT)": "Oxidos de nitrogeno NOx(GT)",
    "PT08.S3(NOx)": "Sensor PT08.S3(NOx)",
    "NO2(GT)": "Dioxido de nitrogeno NO2(GT)",
    "PT08.S4(NO2)": "Sensor PT08.S4(NO2)",
    "PT08.S5(O3)": "Sensor PT08.S5(O3)",
    "T": "Temperatura",
    "RH": "Humedad relativa",
    "AH": "Humedad absoluta",
    "hour": "Hora del dia",
    "month": "Mes",
    "day_of_week": "Dia de la semana",
    "periodo_dia": "Periodo del dia",
}

FEATURE_HELP = {
    "PT08.S1(CO)": "Respuesta del sensor relacionado con monoxido de carbono.",
    "C6H6(GT)": "Concentracion de benceno registrada en la medicion.",
    "PT08.S2(NMHC)": "Respuesta del sensor relacionado con hidrocarburos no metanicos.",
    "NOx(GT)": "Concentracion de oxidos de nitrogeno.",
    "PT08.S3(NOx)": "Respuesta del sensor relacionado con oxidos de nitrogeno.",
    "NO2(GT)": "Concentracion de dioxido de nitrogeno.",
    "PT08.S4(NO2)": "Respuesta del sensor relacionado con dioxido de nitrogeno.",
    "PT08.S5(O3)": "Respuesta del sensor relacionado con ozono.",
    "T": "Temperatura ambiental registrada por el sistema.",
    "RH": "Porcentaje de humedad relativa.",
    "AH": "Cantidad de humedad absoluta.",
    "hour": "Hora de la medicion en formato 0 a 23.",
    "month": "Mes de la medicion en formato 1 a 12.",
    "day_of_week": "Dia de la semana: 0 = lunes y 6 = domingo.",
    "periodo_dia": "Parte del dia a la que pertenece la medicion.",
}


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data():
    raw_data = urllib.request.urlopen(DATA_URL).read()
    zip_file = zipfile.ZipFile(io.BytesIO(raw_data))
    df = pd.read_csv(zip_file.open("AirQualityUCI.csv"), sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].replace(-200, np.nan)

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek

    bins = [-1, 5, 11, 17, 23]
    labels = ["Madrugada", "Manana", "Tarde", "Noche"]
    df["periodo_dia"] = pd.cut(df["hour"], bins=bins, labels=labels).astype("object")

    df = df.drop(columns=["Date", "Time", "datetime", "NMHC(GT)"])
    df = df.dropna(subset=[TARGET]).copy()
    return df


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=0.95, random_state=42)),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=800,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    return model


def build_feature_metadata(X):
    metadata = []
    for feature in NUMERIC_FEATURES:
        values = X[feature].dropna()
        metadata.append(
            {
                "name": feature,
                "label": FEATURE_LABELS[feature],
                "help": FEATURE_HELP[feature],
                "input_type": "number",
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        )

    for feature in CATEGORICAL_FEATURES:
        values = X[feature].dropna().astype(str)
        options = sorted(values.unique().tolist())
        metadata.append(
            {
                "name": feature,
                "label": FEATURE_LABELS[feature],
                "help": FEATURE_HELP[feature],
                "input_type": "select",
                "options": options,
                "default": values.mode().iloc[0],
            }
        )
    return metadata


def train_and_save(output_dir="."):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    metadata = {
        "dataset": "Air Quality Dataset",
        "source": DATA_URL,
        "target": TARGET,
        "target_description": "Concentracion real de monoxido de carbono en el aire.",
        "model": "MLPRegressor",
        "scenario": "PCA",
        "transformation": "Imputacion, codificacion, escalado y PCA al 95% de varianza",
        "pca_components": int(model.named_steps["pca"].n_components_),
        "features": ALL_FEATURES,
        "feature_metadata": build_feature_metadata(X),
        "metrics": metrics,
    }

    joblib.dump(model, output_path / "model.pkl")
    (output_path / "feature_info.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    info = train_and_save()
    print("Modelo guardado en model.pkl")
    print("Modelo:", info["model"])
    print("Escenario:", info["scenario"])
    print("Componentes PCA:", info["pca_components"])
    print("MAE:", round(info["metrics"]["mae"], 4))
    print("R2:", round(info["metrics"]["r2"], 4))
