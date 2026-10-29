from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "data" / "housing.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_FILE = MODELS_DIR / "model.pkl"
PIPELINE_FILE = MODELS_DIR / "pipeline.pkl"

TARGET = "median_house_value"
CAT_ATTRIBS = ["ocean_proximity"]
OCEAN_PROXIMITY_VALUES = ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
