M4 - Multi-Period LLM Query Routing (Team 32 T)
================================================
M3 mathematical formulation and model pool retained. Four categories:
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

OPERATING SETTINGS
alpha=.70 (per-model cap), lambda=10 (loss units/USD), beta=.02 (loss units per
unit allocation change). These are transparent choices, not learned preferences.
The alpha cap does not imply a 30% total-system cost saving. The switching term
penalizes volume changes, including forced demand changes, not only share changes.
Continuous volumes are expected throughput; independent integer rounding is unsafe.
The model selects one common allocation, with budgets enforced in every sampled
scenario. It is not a scenario-dependent recourse policy or a future guarantee.

RESULTS AND VERIFICATION
Base objective 167.145452652979; expected calibration score .773131245378;
heldout weighted expected score .753521126761. Not live deployment accuracy.
48 routing variables, 72 auxiliaries, 112 constraints; Gurobi/HiGHS agree.
results/ contains all allocations, budgets, scenario coefficients, split IDs, LP,
solution, objective checks, sensitivity preview and descriptive oracle evaluation.
The oracle knows each observed answer and ignores constraints: quality bound only.
Do not interpret the oracle gap as regret in the complete cost-sensitive objective.
Beta=0 gives the same allocation here; the base case already achieves the minimum
variation imposed by changing demand. No improvement from beta is claimed.

DELIVERY
Report must be 3-5 pages and contain a working staff-accessible code/data link.
This ZIP is the concrete code/data package to publish after final user approval.
The report's hosting line must be replaced with the verified URL before submission.
No external upload/submission is performed by these scripts.
M3's inaccessible repository URL is not used as a working delivery link.
AI assistance was used for document review, implementation, checks and drafting.
The team should review and understand the work before submission.

REFERENCES
See REFERENCES.txt for full citations, original download, dataset DOI, provenance,
and underlying MMLU/MBPP/WinoGrande attribution; references.bib provides BibTeX.
