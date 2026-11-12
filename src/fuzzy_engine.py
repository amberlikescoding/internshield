"""Member B: a compact zero-order Sugeno fuzzy system (no extra library)."""
import math

# Rows = model low/medium/high; columns = rule low/medium/high.
# Each cell is the output singleton of one IF model AND rule THEN risk rule.
CONSEQUENTS = ((10, 45, 85), (40, 60, 85), (70, 80, 95))
INCOMPLETE_CONSEQUENT = 55


def memberships(value: float) -> dict:
    """Overlapping shoulder/triangle sets on 0..100; weights sum to one."""
    return {
        "low": max(0.0, (50 - value) / 50),
        "medium": max(0.0, 1 - abs(value - 50) / 50),
        "high": max(0.0, (value - 50) / 50),
    }


def calculate_risk(fraud_score: float, rule_score: float, evidence_completeness: float) -> dict:
    for name, value, limit in (("fraud_score", fraud_score, 1),
                               ("rule_score", rule_score, 100),
                               ("evidence_completeness", evidence_completeness, 1)):
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"{name} must be between 0 and {limit}.")
    model = memberships(fraud_score * 100)
    rules = memberships(rule_score)
    trace = []
    for i, (model_name, model_weight) in enumerate(model.items()):
        for j, (rule_name, rule_weight) in enumerate(rules.items()):
            strength = min(model_weight, rule_weight)  # fuzzy AND
            if strength:
                trace.append({"rule": f"model {model_name} AND warnings {rule_name}",
                              "strength": strength, "consequent": CONSEQUENTS[i][j]})
    # Missing evidence supplies a medium caution rule. It can raise a low score,
    # but must never dilute strong warning/model evidence.
    incomplete = 1 - evidence_completeness
    if incomplete:
        trace.append({"rule": "evidence incomplete", "strength": incomplete,
                      "consequent": INCOMPLETE_CONSEQUENT})
    base_trace = [item for item in trace if item["rule"] != "evidence incomplete"]
    base_score = sum(t["strength"] * t["consequent"] for t in base_trace) / sum(t["strength"] for t in base_trace)
    fuzzy_score = sum(t["strength"] * t["consequent"] for t in trace) / sum(t["strength"] for t in trace)
    score = round(max(base_score, fuzzy_score), 1)  # explicit caution guard
    level = "low" if score < 35 else "medium" if score < 65 else "high"
    explanation = (f"Model signal {fraud_score:.1%}; rule severity {rule_score}/100; "
                   f"evidence completeness {evidence_completeness:.0%}. "
                   "Fuzzy AND uses minimum; outputs use a weighted average. "
                   "Missing evidence cannot reduce risk. This is a risk index, not a probability.")
    return {"risk_score": score, "risk_level": level, "explanation": explanation,
            "memberships": {"model": model, "rules": rules}, "trace": trace}
