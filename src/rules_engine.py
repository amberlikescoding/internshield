"""Member B: explicit facts and IF-THEN rules, with no hidden scoring."""
import re

FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me"}
# ID, evidence key, expected value, severity, weight, explanation
RULES = [
    ("R01", "asks_for_fee", True, "high", 65, "An upfront fee was requested."),
    ("R02", "generic_email", True, "low", 10, "Recruiter uses a common personal email provider."),
    ("R03", "interview_occurred", False, "medium", 20, "No interview occurred before the offer."),
    ("R04", "recruiter_confirmed", False, "high", 60, "Independent company contact did not confirm the recruiter."),
    ("R05", "company_verified", False, "medium", 35, "Independent checks did not verify the stated company."),
    ("R06", "requests_secrets", True, "high", 70, "The offer requests a password, OTP, or PIN."),
]


def evaluate_rules(offer_data: dict) -> dict:
    facts = {key: offer_data.get(key) for key in (
        "asks_for_fee", "interview_occurred", "company_verified", "recruiter_confirmed")}
    email = str(offer_data.get("recruiter_email", "")).strip().lower()
    facts["generic_email"] = email.rsplit("@", 1)[-1] in FREE_EMAIL_DOMAINS if "@" in email else None
    # Narrow phrase patterns avoid flagging every mention of security terms.
    # Regex still misses paraphrases and some negations; the trace states it is a text cue.
    text = str(offer_data.get("offer_text", "")).lower()
    secret_pattern = r"\b(?:send|share|provide|submit)\s+(?:(?:us|your|the)\s+){0,2}(?:otp|password|pin)\b"
    negated_pattern = r"\b(?:never|not|don't|do not)\s+(?:send|share|provide|submit)\s+(?:(?:us|your|the)\s+){0,2}(?:otp|password|pin)\b"
    cleaned = re.sub(negated_pattern, "", text)
    facts["requests_secrets"] = bool(re.search(secret_pattern, cleaned))
    flags = []
    score = 0
    for rule_id, key, expected, severity, weight, reason in RULES:
        # Unknown is None, and must not be treated as a negative answer.
        if facts[key] is expected:
            flags.append({"rule_id": rule_id, "severity": severity, "weight": weight, "reason": reason})
            score += weight
    return {"facts": facts, "rule_score": min(score, 100), "flags": flags}
