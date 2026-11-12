import pytest
from src import ml_model
from src.agent import update_evidence
from src.service import assess_offer
from src.report import make_report

TEXT = ("We offer a Python software engineering internship with mentoring, "
        "testing and development tasks. An interview is part of the selection process.")


def test_pipeline_and_manual_check_cycle():
    offer = {"offer_text": TEXT, "recruiter_email": "careers@example.com"}
    evidence = {"asks_for_fee": False, "interview_occurred": True}
    first = assess_offer(offer, evidence)
    assert 0 <= first["model"]["fraud_score"] <= 1
    assert first["agent"]["check_id"] == "verify_company"
    evidence = update_evidence(evidence, "verify_company", True)
    assert assess_offer(offer, evidence)["agent"]["check_id"] == "confirm_recruiter"
    evidence = update_evidence(evidence, "confirm_recruiter", True)
    final = assess_offer(offer, evidence)
    assert final["agent"]["check_id"] is None
    assert final["evidence_completeness"] == 1
    assert "Not a fraud verdict" in make_report(final, [])


def test_fee_scenario_is_high_and_stops():
    result = assess_offer({"offer_text": TEXT, "recruiter_email": "x@gmail.com"},
                          {"asks_for_fee": True, "interview_occurred": False})
    assert result["risk"]["risk_level"] == "high"
    assert result["agent"]["check_id"] is None


def test_empty_or_unrecognised_text_rejected():
    for text in ["", "   ", "zzzzqxxvvv"]:
        with pytest.raises(ValueError):
            ml_model.analyze_text(text)


def test_missing_model_actionable_error(monkeypatch, tmp_path):
    ml_model.load_model.cache_clear()
    monkeypatch.setattr(ml_model, "MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="train_model.py"):
        ml_model.analyze_text(TEXT)
    ml_model.load_model.cache_clear()
