"""Run once after downloading EMSCAD. No network needed during training."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from src.ml_model import MODEL_PATH, ROOT


def train(csv_path: Path):
    df = pd.read_csv(csv_path)
    columns = ["title", "company_profile", "description", "requirements", "benefits"]
    missing = set(columns + ["fraudulent"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    text = df[columns].fillna("").astype(str).agg(" ".join, axis=1)
    # Collapse whitespace/case before detecting identical documents.
    text = text.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    data = pd.DataFrame({"text": text, "label": df["fraudulent"]})
    data = data[data["text"].str.len() > 0]
    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    # Conflicting labels for the exact same text cannot both be correct.
    conflicts = data.groupby("text")["label"].nunique()
    conflict_texts = conflicts[conflicts > 1].index
    data = data[~data["text"].isin(conflict_texts)]
    data = data.drop_duplicates("text")
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42,
        stratify=data["label"],
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=12000,
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        # Equal priors reduce the dominance of the much larger legitimate class.
        # This changes the score: it is not a calibrated real-world probability.
        ("nb", MultinomialNB(alpha=0.5, fit_prior=False)),
    ])
    model.fit(X_train, y_train)  # TF-IDF learns vocabulary/IDF from training only.
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, labels=[0, 1],
                                   target_names=["legitimate", "fraudulent"],
                                   output_dict=True, zero_division=0)
    result = {
        "source": "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction",
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "raw_rows": len(df), "unique_usable_rows": len(data),
        "conflicting_texts_removed": len(conflict_texts),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "random_state": 42, "decision_threshold": 0.5,
        "confusion_matrix_rows_actual_columns_predicted_0_1": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": report,
        "limitations": ["English historical job postings are a proxy for internship offers.",
                        "Near duplicates may remain after exact deduplication.",
                        "Equal class priors; fraud scores are not calibrated probabilities.",
                        "Single held-out split, no hyperparameter search; not a production evaluation."],
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    (ROOT / "docs" / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(classification_report(y_test, predictions, target_names=["legitimate", "fraudulent"], zero_division=0))
    print("Confusion matrix [legitimate, fraudulent]:", confusion_matrix(y_test, predictions).tolist())
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fake_job_postings.csv")
    args = parser.parse_args()
    if not args.data.exists():
        parser.error("CSV missing. Follow data/README.md, then retry.")
    train(args.data)
