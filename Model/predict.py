import os
import pickle
import pandas as pd
import gdown

MODEL_VERSION = "1.0.0"

REDUCED_MODEL_URL = "https://drive.google.com/uc?id=1o0dPggLeTrdVy7dESmsvP6IDk_Q5SeGF"
THRESHOLD_URL = "https://drive.google.com/uc?id=1poS0OD5Z6lhl1pfgg1AyG_GmHGEweYu8"

MODEL_PATH = "reduced_rf_model.pkl"
THRESHOLD_PATH = "screening_threshold.pkl"

def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path} from Google Drive...")
        gdown.download(url, path, quiet=False)
        print(f"{path} downloaded successfully.")

def load_artifacts():
    download_file(REDUCED_MODEL_URL, MODEL_PATH)
    download_file(THRESHOLD_URL, THRESHOLD_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)

    return model, threshold

model, threshold = load_artifacts()

def predict_output(input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])
    prob = model.predict_proba(df)[:, 1][0]
    return prob
