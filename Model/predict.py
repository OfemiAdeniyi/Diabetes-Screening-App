import pickle
import pandas as pd

# Load trained screening model
with open("Model/rf_screening_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load screening threshold
with open("Model/screening_threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

#MLFlow
MODEL_VERSION = '1.0.0'


def predict_output(DiabetesScreeningInput: dict):
    input_df = pd.DataFrame([DiabetesScreeningInput])
    prob = model.predict_proba(input_df)[:, 1][0]
    return prob
