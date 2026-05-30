from functools import lru_cache
from pathlib import Path

from catboost import CatBoostClassifier

from preprocessing import load_metadata, preprocess_inference_record


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.cbm"
METADATA_PATH = ARTIFACTS_DIR / "preprocessing_config.json"


@lru_cache(maxsize=1)
def load_model_bundle() -> tuple[CatBoostClassifier, dict]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python train.py` before starting the API."
        )

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    metadata = load_metadata(METADATA_PATH)
    return model, metadata


def predict_startup(record: dict) -> dict:
    model, metadata = load_model_bundle()
    features = preprocess_inference_record(record, metadata)

    success_probability = float(model.predict_proba(features)[0][1])
    prediction = int(success_probability >= 0.5)

    return {
        "prediction": prediction,
        "predicted_label": "success" if prediction == 1 else "failure",
        "success_probability": success_probability,
        "failure_probability": 1.0 - success_probability,
    }
