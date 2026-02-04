import os
import requests
import joblib
import pandas as pd

MODEL_VERSION = "1.0.0"

REDUCED_MODEL_URL = "https://drive.google.com/uc?id=XXX&export=download"
THRESHOLD_URL = "https://drive.google.com/uc?id=YYY&export=download"

MODEL_PATH = "reduced_rf_model.pkl"
THRESHOLD_PATH = "screening_threshold.pkl"

def download_file(url, path):
    if not os.path.exists(path):
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)

download_file(REDUCED_MODEL_URL, MODEL_PATH)
download_file(THRESHOLD_URL, THRESHOLD_PATH)

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

def predict_output(input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])
    prob = model.predict_proba(df)[:, 1][0]
    return prob

