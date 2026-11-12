# Validation results — 12 November 2026

Command: `python -m pytest -q`

Result: **119 passed**, including parameterized membership checks. One warning came from a preinstalled external pytest plugin (`anyio`), not project code. The tests cover rule flags, simple negations, fuzzy calculation and missing-evidence guard, UCS optimality against exhaustive enumeration for all 16 coverage states, stop policies, unknown-answer handling, a real-model integration cycle, missing model/input errors, and Streamlit form/check/session-state interaction.

Test environment: Python 3.12.14, Linux. Installed project packages: streamlit 1.67.0, scikit-learn 1.8.0, pandas 2.2.3, joblib 1.5.3, pytest 9.2.0.

Windows/macOS instructions are supplied but were not executed on those operating systems. Browser visual review was not performed; Streamlit's AppTest executed the UI flow. Tests check implementation behaviour, not real-world scam-detection validity.

## Held-out text-model results

| Metric | Fraud class |
|---|---:|
| Precision | 25.3% |
| Recall | 80.4% |
| F1 | 0.385 |

Confusion matrix: TN=2693, FP=339, FN=28, TP=115.

The model catches 115 of 143 fraudulent examples but also flags 339 legitimate examples. Its fraud precision is low: most model flags in this held-out split are false positives. This is an intentionally simple baseline with equal priors, not a reliable standalone detector. We retained the result rather than tuning on the held-out test set. A later improvement should use a separate validation set to compare priors/parameters and reserve an untouched test set.

See evaluation.json for the exact split, class counts, source hash and both classes' metrics. Hand-written demo cases and final fuzzy scores are not a measured real-world evaluation.
