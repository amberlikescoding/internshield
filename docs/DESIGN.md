# Design decisions

## Knowledge rules

Facts use `True`, `False`, or `None` for unknown. A missing evidence key means not attempted; a present key with `None` means attempted but unresolved. Neither triggers a negative-fact rule. Email and secret-request cues are derived from the input text; company name is context only.

| ID | Condition | Weight | Severity |
|---|---|---:|---|
| R01 | Upfront fee requested | 65 | High |
| R02 | Common personal email provider | 10 | Low |
| R03 | Interview did not occur | 20 | Medium |
| R04 | Independent contact did not confirm recruiter | 60 | High |
| R05 | Company could not be verified | 35 | Medium |
| R06 | Narrow text pattern requesting password/OTP/PIN | 70 | High |

Sum weights and cap at 100. These constants are a small educational knowledge base. They are not learned from the dataset. R06 handles a few simple negations but is not a general language-understanding system; inspect its trace and do not assume it handles all contexts.

## Fuzzy inference

Inputs: model score × 100, rule severity 0–100, evidence completeness 0–1. For signal x on 0–100:

- Low: `max(0, (50-x)/50)`
- Medium: `max(0, 1-abs(x-50)/50)`
- High: `max(0, (x-50)/50)`

At x=25, Low=0.5, Medium=0.5, High=0. At x=50, Medium=1. These are shoulder/triangle membership functions, not probabilities.

The nine model/rule combinations use fuzzy AND = minimum. Output singletons:

| Model / warnings | Low | Medium | High |
|---|---:|---:|---:|
| Low | 10 | 45 | 85 |
| Medium | 40 | 60 | 85 |
| High | 70 | 80 | 95 |

Each rule fires to a degree w. Sugeno output = sum(w × singleton) / sum(w). An extra incomplete-evidence rule has strength `1-completeness` and singleton 55. We compute scores with/without that rule and take the maximum: this explicit conservative guard prevents incomplete evidence from diluting strong danger signals. This is a fuzzy system with an additional guard, not a pure unmodified Sugeno output.

Risk categories after rounding to one decimal: Low <35, Medium 35–<65, High ≥65. They are human-readable bins over a fuzzy risk index. Membership/output constants are centralized in `src/fuzzy_engine.py`; category cutoffs are in `calculate_risk`.

Example: model=0.25, warnings=25, completeness=1. Four rules fire with strength 0.5, producing (10+45+40+60)/4 = 38.75 → 38.8, Medium. With no known evidence, the extra 55 rule gives 44.2, Medium.

## UCS problem definition

State is a set of attempted fact keys, not a set of companies or risk levels. An action adds one fact key. All actions have positive, illustrative effort costs:

| Action | Evidence added | Cost | Prerequisite attempted fact |
|---|---|---:|---|
| ask_fee | asks_for_fee | 1 | None |
| ask_interview | interview_occurred | 1 | None |
| verify_company | company_verified | 2 | None |
| confirm_recruiter | recruiter_confirmed | 2 | company_verified |
| independent_recruiter_check | recruiter_confirmed | 5 | None |

The company prerequisite means its check was attempted, not that it succeeded. It models already having done the initial company research. This simplification is explicit.

The normal goal is all four checks attempted. From no evidence, the cheapest plan costs 6; directly checking the recruiter costs 5 and leads to total cost 9. UCS explores by cumulative cost using a min-heap and a best-cost table. A counter breaks ties deterministically. It stops when a goal is popped, so it returns the cheapest coverage path on this graph.

The planner does not predict answers. After a real manual answer, update evidence and replan. A known fee request or failed recruiter confirmation stops further investigation early and displays caution. That is a separate policy, not a hypothetical outcome the search assumes. Other warnings remain visible even if the agent continues. Unknown answers count as attempted but do not raise completeness.

Intelligent agent description: environment = supplied offer/evidence; percepts = text and answers; actions = suggested checks; goal = cover defined evidence or stop on a critical warning; performance measure = low assigned investigation effort with explicit explanations. It is a simplified, human-in-the-loop agent, not autonomous web research.

## Model and evaluation

We combine title, company profile, description, requirements and benefits. Labels and metadata flags never enter the text. Whitespace and case are normalized; conflicting identical texts are removed and remaining exact duplicates collapsed before splitting. A fixed stratified 80/20 split preserves minority representation. TF-IDF is fitted only within the training pipeline. Parameters are fixed beforehand; held-out data is not used to choose them.

TF-IDF uses English stopwords, up to 12,000 unigrams/bigrams, min document frequency 2 and logarithmic term frequency. MultinomialNB uses smoothing alpha=0.5 and equal class priors. This avoids letting class prevalence dominate as strongly, at the cost of making scores unlike real-world prevalence. We do not claim optimal settings. TF-IDF is a standard practical fractional-count input for this classifier, although the textbook multinomial model is motivated by counts.

Report both classes' precision, recall, F1 and the confusion matrix. Fraud-class recall matters because false negatives are missed warnings; precision matters because false positives can unfairly alarm students. The evaluation concerns the text classifier only, not real-world validity of the complete hybrid system.
