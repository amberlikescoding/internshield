import json
import numpy as np
import pandas as pd
import pytest

import train_model as training
from src import ml_model


def test_disjoint_reproducible_stratified_splits():
    data = pd.DataFrame({"text": [f"Document {i}" for i in range(100)],
                         "label": [0] * 80 + [1] * 20})
    parts = training.split_data(data)
    assert [len(p) for p in parts] == [60, 20, 20]
    assert [int(p.label.sum()) for p in parts] == [12, 4, 4]
    for i, first in enumerate(parts):
        assert first.equals(training.split_data(data)[i])
        for second in parts[i + 1:]:
            assert set(first.text).isdisjoint(second.text)


def test_threshold_policy_matches_exhaustive_choices():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([.1, .4, .4, .8, .7, .6, .3, .9])
    threshold, _ = training.choose_threshold(labels, scores, .5)
    candidates = [training.measure(labels, scores, t) for t in set(scores)]
    eligible = [r for r in candidates if r["recall"] >= .5]
    expected = max(eligible, key=lambda r: (r["precision"], r["recall"], r["threshold"]))
    assert threshold == expected["threshold"]


def test_constant_scores_and_inclusive_boundary():
    threshold, _ = training.choose_threshold([0, 1], [.5, .5])
    assert threshold == .5
    assert training.measure([0, 1], [.2, .5], .5)["confusion_matrix"] == [[1, 0], [0, 1]]


@pytest.mark.parametrize("labels,scores,minimum", [
    ([0, 1], [.2, .7], 0), ([0, 1], [.2, .7], 1.1),
    ([0, 0], [.2, .7], .7), ([0, 1], [.2, np.nan], .7),
    ([0, 1], [.2, 1.1], .7), ([0, 1], [.2], .7),
])
def test_threshold_bad_inputs(labels, scores, minimum):
    with pytest.raises(ValueError):
        training.choose_threshold(labels, scores, minimum)


def make_csv(path):
    records = []
    for label in [0, 1]:
        for i in range(20):
            records.append({"text": ("python mentor project" if label == 0 else "registration payment deposit") + f" document{i}",
                            "label": label})
    pd.DataFrame(records).to_csv(path, index=False)


def test_cleaning_conflicts_duplicates_and_invalid_labels(tmp_path):
    path = tmp_path / "sample.csv"
    make_csv(path)
    original = pd.read_csv(path)
    repeated = original.iloc[[0]].copy()
    repeated["text"] = "  PYTHON   MENTOR PROJECT document0 "
    conflict = original.iloc[[1]].copy()
    conflict["label"] = 1
    pd.concat([original, repeated, conflict]).to_csv(path, index=False)
    cleaned = training.prepare_data(path)
    assert len(cleaned) == 39
    assert not cleaned.text.duplicated().any()
    original.loc[0, "label"] = 2
    original.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Labels"):
        training.prepare_data(path)


def test_training_only_vocabulary_and_preserved_negation():
    model = training.build_model()
    model.fit(["python mentor", "python project", "registration deposit", "registration fee"], [0, 0, 1, 1])
    training.fraud_scores(model, ["unseentermxyz python"])
    assert "unseentermxyz" not in model.named_steps["tfidf"].vocabulary_
    assert model.named_steps["tfidf"].stop_words is None
    assert model.named_steps["nb"].fit_prior is True


def test_training_uses_one_model_and_selects_before_test(tmp_path, monkeypatch):
    csv_path = tmp_path / "examples.csv"
    make_csv(csv_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "MODEL_PATH", tmp_path / "model.joblib")
    events = []
    real_scores, real_choose = training.fraud_scores, training.choose_threshold
    def track_scores(model, texts):
        events.append(("score", len(texts)))
        return real_scores(model, texts)
    def track_choose(*args):
        events.append(("choose", None))
        return real_choose(*args)
    monkeypatch.setattr(training, "fraud_scores", track_scores)
    monkeypatch.setattr(training, "choose_threshold", track_choose)
    result = training.train(csv_path)
    assert events == [("score", 8), ("choose", None), ("score", 8)]
    assert training.MODEL_PATH.exists()
    report = json.loads((tmp_path / "docs/training_results.json").read_text())
    assert report["threshold"] == result["threshold"]
    import joblib
    artifact = joblib.load(training.MODEL_PATH)
    assert artifact["artifact_version"] == 3
    assert artifact["threshold"] == report["threshold"]
    assert artifact["dataset_sha256"] == training.file_hash(csv_path)


def test_saved_threshold_is_used_at_inference(monkeypatch):
    model = training.build_model()
    text = "python mentor project"
    model.fit([text, text, "registration payment fee", "registration payment fee"], [0, 0, 1, 1])
    score = float(training.fraud_scores(model, [text])[0])
    artifact = {"pipeline": model, "threshold": score, "configuration": "test"}
    monkeypatch.setattr(ml_model, "load_model", lambda: artifact)
    assert ml_model.analyze_text(text)["label"] == "fraudulent"
    artifact["threshold"] = 1.0
    assert ml_model.analyze_text(text)["label"] == "legitimate"


def test_invalid_artifact_rejected(tmp_path, monkeypatch):
    import joblib
    path = tmp_path / "invalid.joblib"
    joblib.dump({"artifact_version": 3, "threshold": float("nan")}, path)
    ml_model.load_model.cache_clear()
    monkeypatch.setattr(ml_model, "MODEL_PATH", path)
    with pytest.raises(ValueError, match="threshold"):
        ml_model.load_model()
    ml_model.load_model.cache_clear()
