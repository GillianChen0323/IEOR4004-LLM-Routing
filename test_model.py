"""Small hand-checkable cases plus independent solver verification (HiGHS).
Every expected number below can be derived by hand; the derivation is in the comments."""
import numpy as np
from router_model import solve


def toy1(d):
    # one category, model A (q=.9, $.2) and model B (q=.5, $.1); reference model for the budget = A
    return dict(q=np.array([[[.9], [.5]]]), c=np.array([[[.2], [.1]]]), d=np.array([d], float),
                p=np.array([1.]), qh=np.array([[.9, .5]]), ch=np.array([[.2, .1]]),
                cats=['toy'], models=['A', 'B'], ref=0)


# ---------------- total budget (M4 default)
P = toy1([10, 10])
r, x = solve(P, 1, 0, 0, crosscheck=True)            # cap = 1 x 10 x .2 = $2: all A fits; loss = .1 x 20
assert np.allclose(x[0, 0], 10) and np.allclose(x[0, 1], 0) and abs(r['objective'] - 2) < 1e-7
r, x = solve(P, .75, 0, 0, crosscheck=True)           # cap $1.5: .2a + .1(10-a) <= 1.5 -> a = 5; loss = (.5+2.5) x 2
assert np.allclose(x, 5) and abs(r['objective'] - 6) < 1e-7
assert abs(r['binding_budgets'][0]['shadow_price_loss_per_usd'] - 4) < 1e-6   # (.5-.1)/(.2-.1) = 4 per $
r, x = solve(P, 1, 10, 0, crosscheck=True)            # lambda=10: A costs .1+2 = 2.1/prompt, B .5+1 = 1.5 -> all B
assert np.allclose(x[0, 1], 10) and abs(r['objective'] - 30) < 1e-7
r, x = solve(P, .4, 0, 0)                             # cheapest plan costs $1.0 > cap $0.8 -> infeasible
assert r['status'] == 3 and x is None
r, x = solve(P, 0, 0, 0); assert r['status'] == 3 and x is None
print('PASS: total budget - accuracy optimum, binding budget + shadow price, price tradeoff, infeasibility; HiGHS cross-checks.')

# ---------------- M3 per-model budget (kept for comparison)
r, x = solve(P, .5, 0, 0, crosscheck=True, budget_mode='per_model', switch_mode='volume')
assert np.allclose(x, 5) and abs(r['objective'] - 6) < 1e-7
print('PASS: M3 per-model budget.')

# ---------------- switching: demand grows 10 -> 20 while the routing rule stays "all A"
P = toy1([10, 20])
r, x = solve(P, 1, 0, .2, crosscheck=True, budget_mode='per_model', switch_mode='volume')
assert np.allclose(x[0, 0], [10, 20]) and abs(r['objective'] - 5) < 1e-7   # M3: forced +10 volume is penalized: 3 + .2 x 10
r, x = solve(P, 1, 0, .2, crosscheck=True)
assert np.allclose(x[0, 0], [10, 20]) and abs(r['switching_measure']) < 1e-9 and abs(r['objective'] - 3) < 1e-7
print('PASS: share-based switching does not penalize demand growth (M3 volume version does).')

# ---------------- share switching trades accuracy for stability
# Categories X, Y; models A, B. X: A (q=1, $1), B (q=0, $0). Y: A (q=1, $1), B (q=1, $0) -> Y always uses B.
# Demand X = [10, 10], Y = [10, 30]; cap = .25 x (all-A spend) = $5 in t1 and $10 in t2.
# t1: X can afford only 5 prompts on A (share .5). t2: X could use A fully (share 1).
# Raising X's A-share in t2 by delta saves 10*delta errors and costs beta * 10 * 2*delta in switching.
P = dict(q=np.array([[[1.], [0.]], [[1.], [1.]]]), c=np.array([[[1.], [0.]], [[1.], [0.]]]),
         d=np.array([[10., 10.], [10., 30.]]), p=np.array([1.]), qh=np.zeros((2, 2)), ch=np.zeros((2, 2)),
         cats=['X', 'Y'], models=['A', 'B'], ref=0)
r, x = solve(P, .25, 0, 0, crosscheck=True)           # beta=0: share .5 -> 1, 5 errors, switching = 10 x (.5+.5) = 10
assert abs(r['objective'] - 5) < 1e-7 and abs(r['switching_measure'] - 10) < 1e-7
r, x = solve(P, .25, 0, .2, crosscheck=True)          # beta=.2 < .5: still switch; 5 + .2 x 10 = 7
assert abs(r['objective'] - 7) < 1e-7 and np.allclose(x[0, 0], [5, 10])
r, x = solve(P, .25, 0, 1, crosscheck=True)           # beta=1 > .5: keep share .5; 5 + 5 errors = 10, no switching
assert abs(r['objective'] - 10) < 1e-7 and np.allclose(x[0, 0], [5, 5]) and abs(r['switching_measure']) < 1e-9
print('PASS: share-based switching penalty - beta threshold .5 reproduced (objective 5 / 7 / 10).')

# Baseline objective must use the selected switching definition.
from router_model import switching_measure
U = np.array([[[10., 20.], [0., 0.]]])
d = np.array([[10., 20.]])
assert switching_measure(U,d,'volume') == 10
assert switching_measure(U,d,'share') == 0
V = np.array([[[10., 0.], [0., 10.]]])
assert switching_measure(V,np.array([[10.,10.]]),'share') == 20
# Ten units move A->B, while L1 counts both the departure and arrival.
assert switching_measure(V,np.array([[10.,10.]]),'share')/2 == 10
print('PASS: baseline volume penalty and L1 versus minimum redistribution semantics.')

# Validate the feasibility bracket stated in the report using the actual frozen data.
import json
from pathlib import Path
from router_model import parameters
P = parameters(json.loads((Path(__file__).parent/'config.json').read_text()))
for alpha, expected in [(0.03,3),(0.034,2),(0.035,2)]:
    result, _ = solve(P,alpha,10,.1)
    assert result['status'] == expected
print('PASS: actual-data budget bracket (.030 infeasible, .034/.035 feasible).')
