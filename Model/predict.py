import os
import requests
import pickle
import pandas as pd

# Version of your ML model
MODEL_VERSION = "1.0.0"

# Google Drive download links (replace with your actual IDs)
REDUCED_MODEL_URL = "https://drive.google.com/uc?id=1o0dPggLeTrdVy7dESmsvP6IDk_Q5SeGF&export=download"
THRESHOLD_URL = "https://drive.google.com/uc?id=1poS0OD5Z6lhl1pfgg1AyG_GmHGEweYu8&export=download"

# Local paths for downloaded files
MODEL_PATH = "reduced_rf_model.pkl"
THRESHOLD_PATH = "screening_threshold.pkl"


def download_file(url: str, path: str):
    """Download a file from a URL if it doesn't already exist locally."""
    if not os.path.exists(path):
        print(f"Downloading {path} from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"{path} downloaded successfully.")


def load_artifacts():
    """Download and load model and threshold."""
    # Download files if missing
    download_file(REDUCED_MODEL_URL, MODEL_PATH)
    download_file(THRESHOLD_URL, THRESHOLD_PATH)

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load threshold
    with open(THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)

    return model, threshold


# Load model and threshold at startup
model, threshold = load_artifacts()


def predict_output(input_dict: dict) -> float:
    """
    Predict probability of diabetes for a single patient input.

    Args:
        input_dict (dict): Dictionary with patient features:
            age, gender, smoking_history, bmi, hypertension_bin, heart_disease_bin

    Returns:
        float: Probability of diabetes (0 to 1)
    """
    df = pd.DataFrame([input_dict])
    # predict_proba returns [[prob_class0, prob_class1]], take prob_class1
    prob = model.predict_proba(df)[:, 1][0]
    return prob
