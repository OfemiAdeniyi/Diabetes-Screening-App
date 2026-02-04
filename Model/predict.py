import os
import requests
import joblib
import pandas as pd

MODEL_VERSION = "1.0.0"

REDUCED_MODEL_URL = "https://drive.google.com/uc?id=XXX&export=download"
THRESHOLD_URL = "https://drive.google.com/uc?id=YYY&export=download"

MODEL_PATH = "reduced_rf_model.pkl"
THRESHOLD_PATH = "screening_threshold.pkl"

model = None
threshold = None


def download_file(url: str, path: str):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"{path} downloaded successfully")


def load_artifacts():
    global model, threshold

    download_file(REDUCED_MODEL_URL, MODEL_PATH)
    download_file(THRESHOLD_URL, THRESHOLD_PATH)

    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    print("Model and threshold loaded successfully")


def predict_output(input_dict: dict) -> float:
    if model is None:
        raise RuntimeError("Model not loaded")

    df = pd.DataFrame([input_dict])
    prob = model.predict_proba(df)[:, 1][0]
    return prob
