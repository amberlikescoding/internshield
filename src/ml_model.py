"""Load the text classifier together with its validation-selected cutoff."""
from functools import lru_cache
from pathlib import Path
import math
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "internshield_text.joblib"


@lru_cache(maxsize=1)
def load_model():
    # Never load untrusted joblib/pickle artifacts: loading can execute code.
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run python train_model.py")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != 3:
        raise ValueError("Unsupported model format. Retrain with train_model.py.")
    threshold = artifact.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid saved decision threshold. Retrain with train_model.py.")
    if "pipeline" not in artifact or "configuration" not in artifact:
        raise ValueError("Incomplete model artifact. Retrain with train_model.py.")
    return artifact


def analyze_text(offer_text: str) -> dict:
    if not isinstance(offer_text, str) or not offer_text.strip():
        raise ValueError("Please enter non-empty offer text.")
    artifact = load_model()
    model = artifact["pipeline"]
    features = model.named_steps["tfidf"].transform([offer_text])
    if features.nnz == 0:
        raise ValueError("No recognised terms. Enter a fuller English or Hinglish recruitment message.")
    column = list(model.classes_).index(1)
    score = float(model.named_steps["nb"].predict_proba(features)[0][column])
    threshold = float(artifact["threshold"])
    return {"label": "fraudulent" if score >= threshold else "legitimate",
            "fraud_score": score, "model_name": "tfidf_multinomial_nb",
            "threshold": threshold, "configuration": artifact["configuration"],
            "calibrated": False, "dataset": artifact.get("dataset", "Indian recruitment synthetic pilot")}
