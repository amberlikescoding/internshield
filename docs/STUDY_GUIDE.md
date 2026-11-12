# Read and practise before the viva

## Study order

1. Run the three examples in DEMO_CASES.md and expand the reasoning trace.
2. Read `src/service.py`: it is the map of the whole pipeline.
3. Read `rules_engine.py`, then `fuzzy_engine.py`; work through the 38.8 example in DESIGN.md by hand.
4. Read `agent.py`; draw states as sets and track the heap on a small example.
5. Read `train_model.py`, then `ml_model.py`; distinguish training from prediction.
6. Read `app.py`; understand why a form submits once and session state survives reruns.
7. Run the tests and explain what at least one test in each area protects.

## Short explanations everyone should know

**What is TF-IDF?** Term frequency measures a term's frequency in a document. Inverse document frequency reduces the influence of terms common across training documents. Scikit-learn applies smoothed IDF and L2 normalization by default; our `sublinear_tf=True` uses 1+log(count). The vocabulary/IDF must never learn from the held-out test set.

**What is Naive Bayes?** Bayes' rule compares P(class) × P(features | class). The naive assumption treats features as conditionally independent given class. This is imperfect for language but yields a fast baseline. Smoothing avoids zero probabilities for unseen feature/class combinations. Scikit-learn handles arithmetic in log space. Naive Bayes is not a full Bayesian Network.

**Why equal priors?** Legitimate examples greatly outnumber fraud examples. Equal priors reduce that prior imbalance. This is an educational modelling choice, not a claim that half of all real offers are fraudulent. The resulting score is not calibrated.

**Why not just accuracy?** An always-legitimate classifier can look accurate on an imbalanced dataset while missing every fraudulent example. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). Define TP as a fraudulent posting correctly flagged.

**Where is knowledge representation?** Named boolean/unknown facts and explicit rules. For example asks_for_fee(offer) implies fee_warning(offer). This is a Python production-rule representation, not a Prolog inference engine or a full theorem prover.

**What makes this fuzzy logic?** A score can partly belong to Low and Medium simultaneously. Rules fire with fractional strengths. This is different from crisp IF score>50 THEN high. Our final category is crisp for readability, but the aggregation uses overlapping memberships.

**What is Sugeno inference?** Each rule contributes a constant output with its firing strength as weight. Divide the weighted output sum by total strength. Our extra caution guard is explicitly separate and prevents missing evidence from lowering risk.

**Why UCS?** Checks have different assigned effort costs. A min-heap explores the least accumulated cost first; positive costs and a finite reachable graph guarantee the first popped goal has minimum cost. BFS minimises steps when edge costs are equal. DFS has no such least-cost guarantee. UCS is equivalent to Dijkstra-style search to a goal.

**Does UCS find the most informative check?** No. It optimises our assigned effort cost to achieve evidence coverage. It does not know outcomes in advance, estimate information gain, or prove the final assessment is correct.

**Why replan?** The coverage plan assumes only that a fact will be attempted. The real answer may trigger the stop policy. The app records it, recalculates risk and asks the agent again.

**What does unknown mean?** Missing key: not attempted. Key=None: attempted but unresolved. Key=True/False: known user-reported answer. Unknown is never silently converted to False.

**Is it an expert system?** A small hybrid prototype: facts/rules act as a knowledge base, the rule/fuzzy functions perform inference, the UI gathers facts, and the trace explains results. It also contains a learned text model and a search planner.

**What are its ethical limitations?** False positives, missed scams, historical/language/domain bias, and unverified user evidence. Avoid personal data, communicate uncertainty and never equate a low score with proven safety.

## Practise these changes yourselves

- Change R02's weight and predict how one example will change before running it.
- Change direct recruiter-check cost from 5 to 1; predict and inspect the new UCS plan.
- Evaluate memberships at 0, 25, 50, 75 and 100 by hand.
- Explain what happens when an offer contains only terms absent from the training vocabulary.
- Explain why training the vectorizer before splitting would leak information.

These exercises make the implementation your own learning rather than a script to memorise. Do not claim authorship or understanding you cannot demonstrate; disclose assistance according to your class rules.
