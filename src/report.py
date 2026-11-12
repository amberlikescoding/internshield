"""A plain-text report can be downloaded or printed without a PDF library."""
import json


def make_report(assessment: dict, history: list) -> str:
    return ("INTERNSHIELD — EDUCATIONAL RISK ASSESSMENT\n"
            "Not a fraud verdict. Manual checks are user-reported; no live lookups.\n"
            "Model score is not a calibrated probability. Low risk does not prove safety.\n\n"
            + json.dumps({"assessment": assessment, "manual_check_history": history}, indent=2))
