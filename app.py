from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
INFO_PATH = BASE_DIR / "feature_info.json"
app = Flask(__name__)

with INFO_PATH.open("r", encoding="utf-8") as file:
    feature_info = json.load(file)
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {item["name"]: round(item["mean"], 4) for item in feature_info["feature_metadata"]}
    if request.method == "POST":
        try:
            input_data = {}
            for item in feature_info["feature_metadata"]:
                name = item["name"]
                value = request.form.get(name, "").strip()
                if value == "":
                    raise ValueError(f"Falta capturar el campo {item['label']}")
                input_data[name] = float(value)
                values[name] = value
            input_df = pd.DataFrame([input_data])
            prediction = float(model.predict(input_df)[0])
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", feature_info=feature_info, prediction=prediction, error=error, values=values)

if __name__ == "__main__":
    app.run(debug=True)