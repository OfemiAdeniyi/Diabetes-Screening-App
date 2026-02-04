import os
import requests
import joblib
import pandas as pd

MODEL_VERSION = "1.0.0"

REDUCED_MODEL_URL = "https://drive.google.com/uc?export=download&id=1o0dPggLeTrdVy7dESmsvP6IDk_Q5SeGF"
THRESHOLD_URL = "https://drive.google.com/uc?export=download&id=1poS0OD5Z6lhl1pfgg1AyG_GmHGEweYu8"

MODEL_PATH = "reduced_rf_model.pkl"
THRESHOLD_PATH = "screening_threshold.pkl"

def download_file(url: str, path: str):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"{path} downloaded")

def load_artifacts():
    download_file(REDUCED_MODEL_URL, MODEL_PATH)
    download_file(THRESHOLD_URL, THRESHOLD_PATH)

    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, threshold

model, threshold = load_artifacts()

def predict_output(input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])
    prob = model.predict_proba(df)[:, 1][0]
    return prob
