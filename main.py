from src.config import MODELS_DIR
from src.train import train


if __name__ == "__main__":
    print("Training model...")
    rmse = train()
    print(f"Test RMSE: ${rmse:,.0f}")
    print(f"Model and pipeline saved to {MODELS_DIR}")
