"""Member C: UCS on a small evidence-gathering graph.

State: frozenset of evidence keys already answered or attempted.
Edges: non-negative effort costs for manual verification actions.
Goal: all four facts attempted, or known fee/recruiter-impersonation warning.
Search plans *coverage*, not unknown outcomes or information gain.
"""
import heapq
from itertools import count

# Some facts have a cheap and an expensive way of checking them. This makes
# UCS choose a least-cost route rather than just follow a hard-coded checklist.
CHECKS = {
    "ask_fee": {"fact": "asks_for_fee", "cost": 1, "requires": set(),
                "question": "Does the offer ask you to pay any fee?",
                "reason": "Clarify whether any fee is requested."},
    "ask_interview": {"fact": "interview_occurred", "cost": 1, "requires": set(),
                      "question": "Did an interview occur before the offer?",
                      "reason": "Clarify the selection process."},
    "verify_company": {"fact": "company_verified", "cost": 2, "requires": set(),
                       "question": "Did your independent checks verify this company?",
                       "reason": "Check an independent company source; existence alone does not verify this offer."},
    "confirm_recruiter": {"fact": "recruiter_confirmed", "cost": 2,
                          "requires": {"company_verified"},
                          "question": "Did an independently found company contact confirm this recruiter?",
                          "reason": "Use independently sourced contact details, not just the offer's contact."},
    "independent_recruiter_check": {"fact": "recruiter_confirmed", "cost": 5, "requires": set(),
                                    "question": "Did independent research and contact confirm this recruiter?",
                                    "reason": "A costlier direct recruiter check is available without prior company evidence."},
}
FACT_KEYS = ("asks_for_fee", "interview_occurred", "company_verified", "recruiter_confirmed")


def stop_reason(evidence: dict):
    if evidence.get("asks_for_fee") is True:
        return "Fee request recorded: stop investigation and review the warning before proceeding."
    if evidence.get("recruiter_confirmed") is False:
        return "Recruiter could not be confirmed: stop and review independent verification."
    if all(key in evidence for key in FACT_KEYS):
        return "All four checks have been answered or attempted. Unknown answers remain unresolved."
    return None


def choose_next_check(evidence_state: dict) -> dict:
    reason = stop_reason(evidence_state)
    if reason:
        return {"check_id": None, "cost": 0, "plan_cost": 0, "plan": [], "reason": reason, "expanded_states": 0}
    start = frozenset(key for key in FACT_KEYS if key in evidence_state)
    serial = count()  # deterministic tie breaking without comparing states
    frontier = [(0, next(serial), start, [])]
    best_cost = {start: 0}
    expanded = 0
    while frontier:
        cost, _, state, path = heapq.heappop(frontier)
        if cost != best_cost[state]:
            continue
        expanded += 1
        if set(FACT_KEYS).issubset(state):
            first = path[0]
            return {"check_id": first, "cost": CHECKS[first]["cost"],
                    "reason": CHECKS[first]["reason"], "plan": path,
                    "plan_cost": cost, "expanded_states": expanded}
        for check_id, check in CHECKS.items():
            if check["fact"] in state or not check["requires"].issubset(state):
                continue
            following = state | {check["fact"]}
            new_cost = cost + check["cost"]
            if new_cost < best_cost.get(following, float("inf")):
                best_cost[following] = new_cost
                heapq.heappush(frontier, (new_cost, next(serial), following, path + [check_id]))
    raise RuntimeError("No verification plan reaches the defined goal.")


def update_evidence(evidence_state: dict, check_id: str, outcome) -> dict:
    if check_id not in CHECKS:
        raise ValueError("Unknown check.")
    if outcome is not None and type(outcome) is not bool:
        raise ValueError("Outcome must be True, False or None (unknown).")
    check = CHECKS[check_id]
    if not check["requires"].issubset(evidence_state):
        raise ValueError("Prerequisite check must be attempted first.")
    if check["fact"] in evidence_state:
        raise ValueError("This evidence was already recorded. Start a new analysis to revise it.")
    # None means attempted but still unknown, so it is not repeatedly requested.
    return {**evidence_state, check["fact"]: outcome}


def evidence_completeness(evidence_state: dict) -> float:
    return sum(evidence_state.get(key) is not None for key in FACT_KEYS) / len(FACT_KEYS)
