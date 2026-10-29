import joblib

from src.config import MODEL_FILE, PIPELINE_FILE


def models_exist():
    return MODEL_FILE.exists() and PIPELINE_FILE.exists()


def load_artifacts():
    return joblib.load(MODEL_FILE), joblib.load(PIPELINE_FILE)


def predict(model, pipeline, houses):
    """Predict median house values for a DataFrame of districts."""
    return model.predict(pipeline.transform(houses))
