# Air Quality Prediction App

Aplicacion Flask para predecir CO(GT) usando el Air Quality Dataset de UCI.

URL oficial: https://archive.ics.uci.edu/ml/datasets/air+quality

URL directa: https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip

## Ejecutar localmente

En PowerShell:

pip install -r requirements.txt; python train_model.py; python app.py

## Render

Build command: pip install -r requirements.txt && python train_model.py
Start command: gunicorn app:app
