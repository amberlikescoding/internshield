"""Check the one-dataset filter, source provenance and PDF integration."""
import json
import fitz
import pandas as pd
import pytest

import download_data
from src.pdf_reader import extract_text_from_pdf
from src.ml_model import analyze_text


def source_rows():
    return pd.DataFrame([
        {"row_id": "a", "text": "Interview tomorrow", "topic": "job_offer", "language": "en", "label": "safe"},
        {"row_id": "b", "text": "Deposit to unlock job", "topic": "job_offer", "language": "hinglish", "label": "scam"},
        {"row_id": "c", "text": "Bank message", "topic": "fake_bank_otp", "language": "en", "label": "safe"},
        {"row_id": "d", "text": "Hindi job", "topic": "job_offer", "language": "hi", "label": "scam"},
        {"row_id": "826f635812fa", "text": "Ambiguous document request", "topic": "job_offer", "language": "en", "label": "scam"},
    ])


def test_filter_keeps_both_labels_only_eligible_languages_and_topic():
    result = download_data.prepare_source(source_rows())
    assert set(result.source_id) == {"a", "b"}
    assert set(result.label) == {0, 1}
    assert list(result.columns) == ["source_id", "text", "label", "language"]
    assert "topic" not in result and "generator" not in result


def test_dataset_export_records_source_and_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "ROOT", tmp_path)
    path = download_data.save_dataset(source_rows().to_csv(index=False).encode())
    assert path.name == "india_jobs.csv"
    metadata = json.loads((path.parent / "dataset_info.json").read_text())
    assert metadata["rows"] == 2 and metadata["synthetic"] is True
    import hashlib
    assert metadata["prepared_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_invalid_source_does_not_overwrite_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "ROOT", tmp_path)
    path = download_data.save_dataset(source_rows().to_csv(index=False).encode())
    original = path.read_bytes()
    bad = source_rows()
    bad.loc[0, "label"] = "unexpected"
    with pytest.raises(ValueError):
        download_data.save_dataset(bad.to_csv(index=False).encode())
    assert path.read_bytes() == original


def test_only_available_training_csv_is_the_single_dataset():
    from train_model import ROOT
    assert {p.name for p in (ROOT / "data").glob("*.csv")} == {
        "internship_offers_train_v2.csv", "internship_offers_holdout_v2.csv"}


def test_holdout_has_no_overlap_with_training_data():
    from train_model import prepare_data, DEFAULT_DATA, DEFAULT_HOLDOUT
    assert set(prepare_data(DEFAULT_DATA).text).isdisjoint(prepare_data(DEFAULT_HOLDOUT).text)


def test_pdf_to_model_prediction():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Software internship with Python development and mentor support.")
        pdf = doc.tobytes()
    result = analyze_text(extract_text_from_pdf(pdf))
    assert result["label"] == ("fraudulent" if result["fraud_score"] >= result["threshold"] else "legitimate")
    assert "Indian" in result["dataset"]
