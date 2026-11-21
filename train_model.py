"""One dataset, one TF-IDF/Naive Bayes model, one validation-selected cutoff,
plus a final check on a separate holdout file that is never used for training or tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "internshield_text.joblib"
DEFAULT_DATA = ROOT / "data" / "internship_offers_train_v2.csv"
DEFAULT_HOLDOUT = ROOT / "data" / "internship_offers_holdout_v2.csv"
SEED = 42
MIN_RECALL = 0.70  # Declared project policy, not an industry standard.


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df["label"].isin([0, 1]).all() or df["text"].isna().any():
        raise ValueError("Labels must be 0 or 1 and text must not be missing.")
    text = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    data = pd.DataFrame({"text": text, "label": df["label"].astype(int)})
    data = data[data["text"].str.len() > 0]
    counts = data.groupby("text")["label"].nunique()
    data = data[~data["text"].isin(counts[counts > 1].index)]
    data = data.drop_duplicates("text").reset_index(drop=True)
    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 5:
        raise ValueError("Need at least five unique examples of each class for splitting.")
    return data


def split_data(data):
    development, test = train_test_split(
        data, test_size=0.20, stratify=data["label"], random_state=SEED)
    train, validation = train_test_split(
        development, test_size=0.25, stratify=development["label"], random_state=SEED)
    return train, validation, test


def build_model():
    return Pipeline([
        # Keep negation words and Hinglish; rare words are retained in this small dataset.
        ("tfidf", TfidfVectorizer(stop_words=None, max_features=5000,
                                  ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        # Learn priors from our near-balanced training partition; no oversampling.
        ("nb", MultinomialNB(alpha=0.5, fit_prior=True)),
    ])


def fraud_scores(model, texts):
    column = list(model.classes_).index(1)
    return model.predict_proba(texts)[:, column]


def measure(labels, scores, threshold):
    predictions = (np.asarray(scores) >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "false_alarms": int(matrix[0, 1]), "missed_fraud": int(matrix[1, 0]),
    }


def choose_threshold(labels, scores, min_recall=MIN_RECALL):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1].")
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both classes, labelled 0 and 1.")
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match the label count.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be between 0 and 1.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    curve = pd.DataFrame({"threshold": thresholds,
                          "precision": precision[:-1], "recall": recall[:-1]})
    eligible = curve[curve["recall"] >= min_recall]
    best = eligible.sort_values(["precision", "recall", "threshold"],
                                ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), curve


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_holdout(model, threshold, holdout_path, training_texts):
    """Score the separate holdout file once, with the cutoff already fixed."""
    holdout = prepare_data(holdout_path)
    overlap = int(holdout["text"].isin(set(training_texts)).sum())
    holdout = holdout[~holdout["text"].isin(set(training_texts))]  # never score a training text
    scores = fraud_scores(model, holdout["text"])
    result = measure(holdout["label"], scores, threshold)
    result.update({"rows": len(holdout), "removed_overlap_with_training": overlap,
                   "file_sha256": file_hash(holdout_path)})
    return result


def train(csv_path, min_recall=MIN_RECALL, holdout_path=None):
    data = prepare_data(csv_path)
    train_set, validation, test = split_data(data)
    model = build_model()
    model.fit(train_set["text"], train_set["label"])
    validation_scores = fraud_scores(model, validation["text"])
    threshold, curve = choose_threshold(validation["label"], validation_scores, min_recall)
    # All design choices/cutoff are fixed before evaluating the test and holdout data.
    test_scores = fraud_scores(model, test["text"])
    holdout_results = None
    if holdout_path is not None:
        holdout_results = evaluate_holdout(model, threshold, holdout_path, data["text"])
    counts = {name: {"total": len(part), "legitimate": int((part.label == 0).sum()),
                     "fraudulent": int((part.label == 1).sum())}
              for name, part in [("train", train_set), ("validation", validation), ("test", test)]}
    summary = {
        "dataset": "Indian internship postings and offers (English/Hinglish) — hand-written synthetic",
        "dataset_sha256": file_hash(csv_path), "script_sha256": file_hash(__file__),
        "seed": SEED, "split_counts": counts, "calibrated": False,
        "threshold": threshold,
        "policy": {"minimum_validation_recall": min_recall,
                   "selection": "Highest validation precision, ties prefer recall then higher cutoff."},
        "validation_results": measure(validation.label, validation_scores, threshold),
        "test_results": measure(test.label, test_scores, threshold),
        "holdout_results": holdout_results,
        "limitations": [
            "Small synthetic dataset written by hand, not a verified real-message benchmark.",
            "A random internal split can share writing styles/templates; exact duplicates are removed, near duplicates may remain.",
            "The holdout file is separate and was never used for training or choosing the cutoff, but it is small and written by the same team/assistant.",
            "Model is uncalibrated and not validated for full offer letters or real-world scam detection.",
        ],
    }
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    curve.to_csv(docs / "validation_pr_curve.csv", index=False)
    (docs / "training_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"artifact_version": 3, "pipeline": model, "threshold": threshold,
                 "configuration": "indian_recruitment_nb", "calibrated": False,
                 "dataset_sha256": summary["dataset_sha256"], "dataset": summary["dataset"]}, MODEL_PATH)
    print(json.dumps(summary, indent=2))
    print("Saved:", MODEL_PATH)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    args = parser.parse_args()
    try:
        train(args.data, holdout_path=args.holdout if args.holdout.exists() else None)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
