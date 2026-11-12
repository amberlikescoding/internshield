"""Member A: convert English offer text into a dataset-based fraud signal."""
from functools import lru_cache
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tfidf_nb_pipeline.joblib"


@lru_cache(maxsize=1)
def load_model():
    # Load only the model produced by our own training script. Pickles are executable.
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run: python train_model.py")
    return joblib.load(MODEL_PATH)


def analyze_text(offer_text: str) -> dict:
    if not isinstance(offer_text, str) or not offer_text.strip():
        raise ValueError("Please enter non-empty offer text.")
    model = load_model()
    if model.named_steps["tfidf"].transform([offer_text]).nnz == 0:
        raise ValueError("No recognised English terms. Enter a fuller English offer.")
    classes = list(model.classes_)
    score = float(model.predict_proba([offer_text])[0][classes.index(1)])
    return {
        "label": "fraudulent" if score >= 0.5 else "legitimate",
        "fraud_score": score,
        "model_name": "tfidf_multinomial_nb",
    }
