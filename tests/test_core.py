import itertools
import math
import pytest
from src.agent import CHECKS, FACT_KEYS, choose_next_check, update_evidence, evidence_completeness
from src.fuzzy_engine import calculate_risk, memberships
from src.rules_engine import evaluate_rules


def test_unknown_does_not_fire_negative_rules():
    result = evaluate_rules({})
    assert result["flags"] == []
    assert result["facts"]["interview_occurred"] is None


def test_all_rules_and_cap():
    result = evaluate_rules({"asks_for_fee": True, "interview_occurred": False,
                             "recruiter_confirmed": False, "company_verified": False,
                             "recruiter_email": "test@gmail.com", "offer_text": "Please share your OTP"})
    assert {x["rule_id"] for x in result["flags"]} == {f"R0{i}" for i in range(1, 7)}
    assert result["rule_score"] == 100


@pytest.mark.parametrize("text", ["Never share your OTP", "Do not send your password", "don't provide your PIN"])
def test_basic_negation(text):
    assert not evaluate_rules({"offer_text": text})["facts"]["requests_secrets"]


@pytest.mark.parametrize("x", range(101))
def test_memberships_cover_input(x):
    values = memberships(x).values()
    assert all(0 <= v <= 1 for v in values)
    assert sum(values) == pytest.approx(1)


def test_fuzzy_worked_example_and_boundaries():
    assert calculate_risk(.25, 25, 1)["risk_score"] == 38.8
    assert calculate_risk(0, 0, 1)["risk_level"] == "low"
    assert calculate_risk(1, 100, 1)["risk_level"] == "high"
    for model, rules in itertools.product([0, .25, .5, .75, 1], [0, 25, 50, 75, 100]):
        known = calculate_risk(model, rules, 1)["risk_score"]
        unknown = calculate_risk(model, rules, 0)["risk_score"]
        assert 0 <= known <= unknown <= 100


@pytest.mark.parametrize("args", [(-1, 0, 0), (1, 101, 0), (1, 0, 2), (math.nan, 0, 0)])
def test_invalid_fuzzy_inputs(args):
    with pytest.raises(ValueError):
        calculate_risk(*args)


def cheapest_by_enumeration(evidence):
    # Independent exhaustive baseline; small enough to enumerate all legal paths.
    state = set(evidence)
    if set(FACT_KEYS) <= state:
        return 0
    costs = []
    for check in CHECKS.values():
        if check["fact"] not in state and check["requires"] <= state:
            costs.append(check["cost"] + cheapest_by_enumeration({**evidence, check["fact"]: None}))
    return min(costs)


def test_ucs_optimal_for_every_coverage_state():
    for bits in itertools.product([False, True], repeat=4):
        evidence = {key: None for key, present in zip(FACT_KEYS, bits) if present}
        plan = choose_next_check(evidence)
        assert plan["plan_cost"] == cheapest_by_enumeration(evidence)
    assert choose_next_check({})["plan_cost"] == 6
    assert "independent_recruiter_check" not in choose_next_check({})["plan"]


def test_ucs_stops_on_actual_critical_answer():
    assert choose_next_check({"asks_for_fee": True})["check_id"] is None
    assert choose_next_check({"recruiter_confirmed": False})["check_id"] is None


def test_unknown_update_progresses_without_claiming_knowledge():
    evidence = update_evidence({}, "ask_fee", None)
    assert evidence_completeness(evidence) == 0
    assert choose_next_check(evidence)["check_id"] != "ask_fee"
    with pytest.raises(ValueError):
        update_evidence(evidence, "ask_fee", False)
    with pytest.raises(ValueError):
        update_evidence({}, "confirm_recruiter", True)
    with pytest.raises(ValueError):
        update_evidence({}, "ask_fee", "yes")
