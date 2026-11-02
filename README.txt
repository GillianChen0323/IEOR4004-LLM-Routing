M4 - Multi-Period LLM Query Routing (Team 32 T)
================================================
M3 model pool, categories and objective retained; budget and switching terms refined. Four categories:
math, coding, commonsense, reasoning. No Accounting category in this release.

SETUP AND RUN
Python 3.12 recommended; install the tested requirements:
  python -m pip install -r requirements.txt
  python router_model.py --out results
  python test_model.py
Gurobi requires a valid license. This 120-variable LP was tested with a restricted
non-production license. HiGHS independently verifies the same LP. No LLM API calls
or API keys are used. Paths are script-relative. Unpack all files with data/ intact.
The notebook wrapper runs this same program, rather than a separate implementation.

DATA
RouterBench 0-shot: https://huggingface.co/datasets/withmartian/routerbench
Pinned original SHA-256 and compact CSV hash are in data/provenance.json.
36,497 source rows; 2,361 selected; 2 exact duplicate prompts removed; 2,359 retained.
M3 leaves benchmark mapping unspecified. M4 explicitly maps:
 math -> mmlu-elementary-mathematics
 coding -> mbpp
 commonsense -> winogrande
 reasoning -> mmlu-formal-logic + mmlu-logical-fallacies
This mapping follows task content. It is not a claim about undocumented teammate code.
The CSV includes all three model scores and costs on the same selected rows, not raw text.
Optional: python rebuild_data.py downloads ~95 MiB, verifies the original hash,
restricts pickle classes, and checks reconstructed rows against the included data.

METHOD
Stratified 70/30 calibration/heldout split, seed 4004: 1,649 / 710 rows.
Five paired bootstrap scenarios use calibration scores and costs only.
Heldout category counts define workload; heldout scores enter post-solve evaluation only.
Four periods; coding receives 45% in period 2, remaining 55% evenly elsewhere.
Other categories are uniform. These are simulated periods, not recorded timestamps.
M3's mixtral-8x7b corresponds to the source column mistralai/mixtral-8x7b-chat.

FORMULATION (M4 = M3 with two refinements; M3 versions kept as options)
budget_mode="total"  : one cap per period and scenario on TOTAL spend,
                       B_ts = alpha * sum_k d_kt * c_k,GPT-4,s  (spend if all traffic went to GPT-4).
                       M3's per-model caps ("per_model") let total spend RISE when alpha tightened.
switch_mode="share"  : penalize change in routing SHARES x_kmt/d_kt, weighted by d_kt
                       (demand-weighted L1, not unique rerouted requests). Half this distance is a
                       minimum redistribution proxy. Volume mode also charges demand changes;
                       beta=0 and .02 gave the same allocation at the tested M3 baseline.
Run the M3 version: solve(P, 0.70, 10, 0.02, budget_mode="per_model", switch_mode="volume")
(reproduces the earlier M4 draft objective 167.145453).

OPERATING SETTINGS
alpha=.30 (total spend <= 30% of all-GPT-4 spend per period), lambda=10 (loss units/USD),
beta=.10 (loss units per demand-weighted L1 routing-change unit). Transparent choices, not fitted.
On this data only coding is cost-sensitive (break-even lambda ~1.8); without a budget,
lambda=10 already cuts spend to ~29% of all-GPT-4, so alpha must be below ~0.33 to bind.
At lambda=0 and 10 the tested allocation is unchanged, not a universal binding-budget rule.
Budget dual multipliers add scarcity cost to lambda; they do not replace it. A shadow
price measures local improvement in the full objective when one budget RHS increases.
Tightening alpha lowers allowed caps, not necessarily the optimizer's actual expected spend.

RESULTS AND VERIFICATION
Base objective 174.956625; expected calibration score .759095; spend $0.391 (24% of all-GPT-4);
heldout weighted expected score .746779 vs .763380 for all-GPT-4. Not live deployment accuracy.
Plan: coding -> GPT-3.5; math, commonsense -> GPT-4; reasoning -> 20% GPT-4 / 80% Mixtral.
beta=0 would move reasoning to 100% GPT-4 in period 2 only (routing-change L1=69.4; half=34.7 redistribution proxy, 2.5 fewer
expected errors); beta=.10 keeps the plan stable.
48 routing variables, 72 auxiliaries, 72 constraints (16 demand, 20 budget, 36 coupling);
Gurobi/HiGHS agree. results/ has allocations, budgets with shadow prices, baselines,
scenario coefficients, split IDs, LP, solution, sensitivity preview and oracle evaluation.
The oracle knows each observed answer and ignores constraints: quality bound only.

DELIVERY
Report must be 3-5 pages and contain a working staff-accessible code/data link.
This ZIP is the concrete code/data package to publish after final user approval.
Repository: https://github.com/GillianChen0323/IEOR4004-LLM-Routing
The revised files must be published and verified there before the report is submitted.
No external upload/submission is performed by these scripts.
M3's inaccessible repository URL is not used as a working delivery link.
AI assistance was used for document review, implementation, checks and drafting.
The team should review and understand the work before submission.

REFERENCES
See REFERENCES.txt for full citations, original download, dataset DOI, provenance,
and underlying MMLU/MBPP/WinoGrande attribution; references.bib provides BibTeX.
