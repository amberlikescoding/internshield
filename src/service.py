"""Small orchestrator shared by the interface and integration tests."""
from src.ml_model import analyze_text
from src.rules_engine import evaluate_rules
from src.fuzzy_engine import calculate_risk
from src.agent import choose_next_check, evidence_completeness


def assess_offer(offer: dict, evidence: dict) -> dict:
    model = analyze_text(offer["offer_text"])
    rules = evaluate_rules({**offer, **evidence})
    completeness = evidence_completeness(evidence)
    risk = calculate_risk(model["fraud_score"], rules["rule_score"], completeness)
    return {"model": model, "rules": rules, "evidence": evidence,
            "evidence_completeness": completeness, "risk": risk,
            "agent": choose_next_check(evidence)}
