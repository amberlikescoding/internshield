# InternShield

A small Python AI class project for three students: explainable risk assessment of internship and job offers. It is a learning prototype, not a service that verifies companies or declares offers genuine/fake.

## Start here

Use **Python 3.12** (tested) on Windows or macOS. Python 3.11 is expected to work but was not tested. Open a terminal in this project directory.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and `.venv\Scripts\activate.bat`, or run `.venv\Scripts\python.exe` in place of `python`. No policy changes required.

Activate on macOS:

```bash
source .venv/bin/activate
```

If your Mac uses `python3`, use it to create the environment; `python` works once it is activated.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The review ZIP includes `models/tfidf_nb_pipeline.joblib`, so it runs without training or a dataset download. Only load our own generated model: joblib files from unknown sources can execute code.

A fresh Git clone does not include the model or raw data. Rebuild them:

```bash
python download_data.py
python train_model.py
python -m streamlit run app.py
```

Training runs locally on CPU. `python -m pytest -q` runs tests; the model must exist for integration/UI tests. Missing-model behaviour is tested separately.

## What happens

1. A Streamlit form collects English offer text, an optional company name/email, and known fee/interview facts.
2. TF-IDF converts text to numeric features. Multinomial Naive Bayes estimates similarity to fraudulent/legitimate dataset examples.
3. Six IF–THEN rules produce visible reasons and severity points.
4. Uniform Cost Search chooses a least-effort plan to cover missing evidence. The user manually performs the next check and records an answer. The agent replans after each answer.
5. A small Sugeno fuzzy system combines model signal and rule severity. Missing evidence adds caution. A documented guard prevents missing evidence from lowering risk.
6. The app shows separate signals, a complete reasoning trace, and a downloadable text report.

No network requests occur in the app. Only the separate dataset downloader uses the internet. No login, database, paid API, Prolog, JavaScript, or custom CSS. Offer details stay in the running Streamlit session unless you download a report; they are not written to disk by the app.

## Files and three-person ownership

| Owner | Files | Concepts to explain |
|---|---|---|
| A | `train_model.py`, `download_data.py`, `src/ml_model.py` | TF-IDF, conditional probability, Naive Bayes, train/test split, precision/recall |
| B | `src/rules_engine.py`, `src/fuzzy_engine.py` | Facts, rules, fuzzy membership, AND, weighted output |
| C | `src/agent.py`, `src/service.py`, `app.py`, `src/report.py` | State graph, UCS, stopping, Streamlit session state, integration |
| All | `tests/`, `docs/`, README | End-to-end demo, limitations and viva |

These are study responsibilities, not a claim that each person has already authored those files. Follow your institution's rules for AI assistance and attribution.

Read [the study guide](docs/STUDY_GUIDE.md), [demo cases](docs/DEMO_CASES.md), and [design decisions](docs/DESIGN.md). Actual held-out results are in [evaluation.json](docs/evaluation.json).

## Scope and limitations

- English historical job postings are a proxy for internship offers. Short messages, other languages and new scam patterns may behave poorly.
- Model labels are dataset classes; its score is not a calibrated real-world probability. Equal class priors intentionally change the score.
- Generic email and absence of an interview are weak signals. No single rule proves fraud.
- Manual verification is user-reported, not independently validated by this app. Website existence does not verify an offer.
- UCS minimises assigned effort costs, not information gain. Its planned transitions assume attempted checks, not known outcomes.
- Fuzzy constants, rule weights and risk category cutoffs are illustrative expert choices, not empirically validated thresholds.
- Low risk does not establish safety. Unknown facts remain unknown; attempting every check does not guarantee complete evidence.
- Exact text duplicates and conflicting labels are removed before splitting, but near duplicates and other dataset biases can remain.

## Repository

Project repository: https://github.com/amberlikescoding/internshield

The source code is public with the owner's approval. The raw dataset and generated model are excluded from Git; follow the training instructions above after cloning.
