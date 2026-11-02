"""Small hand-checkable cases plus independent solver verification."""
import numpy as np
from router_model import solve
P=dict(q=np.array([[[.9],[.5]]]),c=np.array([[[.2],[.1]]]),d=np.array([[10.,10.]]),p=np.array([1.]),qh=np.array([[.9,.5]]),ch=np.array([[.2,.1]]),cats=['toy'],models=['A','B'])
r,x=solve(P,1,0,0,crosscheck=True)
assert np.allclose(x[0,0],10) and np.allclose(x[0,1],0) and abs(r['objective']-2)<1e-7
r,x=solve(P,.5,0,0,crosscheck=True)
assert np.allclose(x,5) and abs(r['objective']-6)<1e-7
r,x=solve(P,1,10,0,crosscheck=True)
assert np.allclose(x[0,1],10) and abs(r['objective']-30)<1e-7
r,x=solve(P,0,0,0);assert r['status']==3 and x is None
print('PASS: accuracy optimum, binding budgets, price tradeoff, infeasibility; 3 HiGHS cross-checks.')
# Unequal demand: switching term must equal the actual adjacent-volume change.
P['d']=np.array([[10.,20.]])
r,x=solve(P,1,0,.2,crosscheck=True)
assert np.allclose(x[0,0],[10,20]) and np.allclose(x[0,1],0)
assert abs(r['allocation_L1_change']-10)<1e-7 and abs(r['objective']-5)<1e-7
print('PASS: unequal-period demand and correctly weighted switching penalty.')
