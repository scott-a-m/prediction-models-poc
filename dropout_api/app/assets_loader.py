import os
import joblib

def load_asset(filename: str, directory: str) -> object:
    base_path = os.path.dirname(os.path.dirname(__file__))
    asset_path = os.path.abspath(os.path.join(base_path, directory, filename))

    print(f"Loading asset from: {asset_path}")

    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Asset not found: {asset_path}")

    return joblib.load(asset_path)

def load_model(model_filename: str, model_dir: str) -> object:
    return load_asset(model_filename, model_dir)

def load_encoders(encoders_filename: str, encoders_dir: str) -> object:
    return load_asset(encoders_filename, encoders_dir)