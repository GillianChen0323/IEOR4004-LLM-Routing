"""M4: M3 algebra and categories; explicit benchmark mapping.
No LLM API calls. All costs and performance are historical benchmark estimates.
Run: python router_model.py --out results
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parent

def parameters(config):
    df=pd.read_csv(ROOT/'data/routerbench_selected.csv')
    models=config['models'];cats=['math','coding','commonsense','reasoning']
    score_cols=models;cost_cols=[m+'|total_cost' for m in models]
    a=df[score_cols].to_numpy(float);c=df[cost_cols].to_numpy(float)
    assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    assert np.isfinite(c).all() and (c>=0).all()
    assert df.sample_id.is_unique
    rng=np.random.default_rng(config['seed']);K=len(cats);M=len(models);T=config['periods'];S=config['scenarios']
    assert T==4 and S>0 and 0<config['calibration_fraction']<1
    assert 0<config['coding_period2_share']<1
    assert set(df.category)==set(cats)
    assert hashlib.sha256((ROOT/'data/routerbench_selected.csv').read_bytes()).hexdigest()==json.loads((ROOT/'data/provenance.json').read_text())['selected_csv_sha256']
    q=np.zeros((K,M,S));cost=q.copy();qh=np.zeros((K,M));ch=qh.copy();d=np.zeros((K,T));split=[];counts=[]
    for k,cat in enumerate(cats):
        rows=df[df.category==cat].copy();ids=rng.permutation(len(rows));n=int(np.floor(len(rows)*config['calibration_fraction']))
        assert 0<n<len(rows), 'Each category needs calibration and heldout rows'
        train=rows.iloc[ids[:n]];test=rows.iloc[ids[n:]]
        for label,part in [('calibration',train),('heldout',test)]:
            split.extend({'sample_id':v,'category':cat,'split':label} for v in part.sample_id)
        counts.append({'category':cat,'total':len(rows),'calibration':len(train),'heldout':len(test)})
        for s in range(S):
            boot=train.iloc[rng.integers(0,len(train),size=len(train))]
            q[k,:,s]=boot[score_cols].mean().to_numpy();cost[k,:,s]=boot[cost_cols].mean().to_numpy()
        qh[k]=test[score_cols].mean().to_numpy();ch[k]=test[cost_cols].mean().to_numpy()
        shares=np.full(T,1/T)
        if cat=='coding':
            spike=config['coding_period2_share'];shares=np.array([(1-spike)/3,spike,(1-spike)/3,(1-spike)/3])
        d[k]=len(test)*shares
    assert np.allclose(d.sum(),sum(x['heldout'] for x in counts))
    return dict(q=q,c=cost,d=d,qh=qh,ch=ch,cats=cats,models=models,counts=counts,split=split,p=np.full(S,1/S))

def solve(P,alpha,lam,beta,out=None,crosscheck=False):
    assert alpha>=0 and lam>=0 and beta>=0
    q,c,d,p=P['q'],P['c'],P['d'],P['p'];K,M,S=q.shape;T=d.shape[1]
    B=alpha*np.einsum('kms,kt->mts',c,d)
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    model=gp.Model('LLM_Routing_M4',env=env);model.Params.Method=1;model.Params.FeasibilityTol=1e-8
    x=model.addVars(K,M,T,lb=0,name='x');wp=model.addVars(K,M,range(1,T),lb=0,name='wp');wm=model.addVars(K,M,range(1,T),lb=0,name='wm')
    coeff=np.einsum('kms,s->km',1-q+lam*c,p)
    model.setObjective(gp.quicksum(coeff[k,m]*x[k,m,t] for k in range(K) for m in range(M) for t in range(T))+beta*gp.quicksum(wp[key]+wm[key] for key in wp),GRB.MINIMIZE)
    for k in range(K):
        for t in range(T):model.addConstr(gp.quicksum(x[k,m,t] for m in range(M))==d[k,t],name=f'demand_{k}_{t}')
    for m in range(M):
        for t in range(T):
            for s in range(S):model.addConstr(gp.quicksum(c[k,m,s]*x[k,m,t] for k in range(K))<=B[m,t,s],name=f'budget_{m}_{t}_{s}')
    for k,m,t in wp:model.addConstr(x[k,m,t]-x[k,m,t-1]==wp[k,m,t]-wm[k,m,t],name=f'change_{k}_{m}_{t}')
    model.optimize()
    result={'status':int(model.Status),'alpha':alpha,'lambda_cost':lam,'beta_switch':beta}
    if model.Status!=GRB.OPTIMAL:
        if out:
            out.mkdir(parents=True,exist_ok=True);model.write(str(out/'model.lp'))
            if model.Status==GRB.INFEASIBLE:model.computeIIS();model.write(str(out/'infeasibility.ilp'))
        model.dispose();env.dispose();return result,None
    X=np.array([[[x[k,m,t].X for t in range(T)] for m in range(M)] for k in range(K)])
    spend=np.einsum('kms,kmt->mts',c,X)
    error=float(np.einsum('kms,kmt,s->',1-q,X,p));api=float(np.einsum('kms,kmt,s->',c,X,p));variation=float(np.abs(np.diff(X,axis=2)).sum())
    objective=error+lam*api+beta*variation
    demand_gap=float(np.max(np.abs(X.sum(axis=1)-d)));budget_excess=float(max(0,(spend-B).max()));min_x=float(X.min())
    assert demand_gap<1e-6 and budget_excess<1e-7 and min_x>=-1e-7
    assert abs(model.ObjVal-objective)<1e-5
    total=float(d.sum());heldout_error=float(np.einsum('km,kmt->',1-P['qh'],X));heldout_cost=float(np.einsum('km,kmt->',P['ch'],X))
    heldout_excess=float(max(0,(np.einsum('km,kmt->mt',P['ch'],X)-np.einsum('mts,s->mt',B,p)).max()))
    result.update(objective=objective,expected_error_count=error,expected_api_usd=api,allocation_L1_change=variation,expected_accuracy=1-error/total,heldout_expected_accuracy=1-heldout_error/total,heldout_expected_api_usd=heldout_cost,heldout_max_excess_over_mean_budget_usd=heldout_excess,total_demand=total,variables=model.NumVars,constraints=model.NumConstrs,checks={'max_demand_residual':demand_gap,'max_calibration_budget_violation_usd':budget_excess,'min_allocation':min_x,'objective_reconstruction_gap':abs(objective-model.ObjVal)})
    # Independent solver verifies the same LP; direct calculations above verify business constraints.
    if crosscheck:
        A=model.getA().toarray();rhs=np.array(model.getAttr('RHS'));senses=np.array(model.getAttr('Sense'));costvec=np.array(model.getAttr('Obj'));eq=senses=='=';ub=senses=='<'
        check=linprog(costvec,A_ub=A[ub],b_ub=rhs[ub],A_eq=A[eq],b_eq=rhs[eq],bounds=(0,None),method='highs')
        assert check.success and abs(check.fun-objective)<1e-5
        result['checks']['independent_HiGHS_objective']=float(check.fun)
    if out:
        out.mkdir(parents=True,exist_ok=True)
        rows=[{'category':P['cats'][k],'model':P['models'][m],'period':t+1,'allocated_prompts':X[k,m,t],'demand':d[k,t],'routing_share':X[k,m,t]/d[k,t]} for k in range(K) for m in range(M) for t in range(T)]
        pd.DataFrame(rows).to_csv(out/'allocations.csv',index=False)
        pd.DataFrame([{'model':P['models'][m],'period':t+1,'scenario':s,'spend_usd':spend[m,t,s],'cap_usd':B[m,t,s],'slack_usd':B[m,t,s]-spend[m,t,s]} for m in range(M) for t in range(T) for s in range(S)]).to_csv(out/'budget_checks.csv',index=False)
        model.write(str(out/'model.lp'));model.write(str(out/'solution.sol'))
        (out/'summary.json').write_text(json.dumps(result,indent=2))
    model.dispose();env.dispose();return result,X

def run(out):
    cfg=json.loads((ROOT/'config.json').read_text());P=parameters(cfg);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(P['counts']).to_csv(out/'split_counts.csv',index=False);pd.DataFrame(P['split']).to_csv(out/'split_manifest.csv',index=False)
    pd.DataFrame([{'category':P['cats'][k],'model':P['models'][m],'scenario':s,'quality':P['q'][k,m,s],'cost_usd':P['c'][k,m,s]} for k in range(len(P['cats'])) for m in range(len(P['models'])) for s in range(cfg['scenarios'])]).to_csv(out/'scenario_parameters.csv',index=False)
    result,X=solve(P,cfg['alpha'],cfg['lambda_cost'],cfg['beta_switch'],out,crosscheck=True)
    if X is None:
        raise RuntimeError(f'Base case has no optimal solution; solver status {result["status"]}')
    # A uniform allocation is feasible for alpha>=1/M; use it as a sanity-check baseline.
    U=np.repeat((P['d']/len(P['models']))[:,None,:],len(P['models']),axis=1)
    uc=float(np.einsum('kms,kmt,s->',1-P['q']+cfg['lambda_cost']*P['c'],U,P['p'])+cfg['beta_switch']*abs(np.diff(U,axis=2)).sum())
    assert cfg['alpha']>=1/len(P['models']) and result['objective']<=uc+1e-6
    cases=[]
    for label,a,l,b in [('lambda0',cfg['alpha'],0,cfg['beta_switch']),('beta0',cfg['alpha'],cfg['lambda_cost'],0),('tight_budget',0.4,cfg['lambda_cost'],cfg['beta_switch']),('loose_budget',1.0,cfg['lambda_cost'],cfg['beta_switch']),('zero_budget',0,cfg['lambda_cost'],cfg['beta_switch'])]:
        rr,_=solve(P,a,l,b);rr['case']=label;cases.append(rr)
    assert cases[-1]['status']==GRB.INFEASIBLE
    checks={'Gurobi_and_HiGHS_agree':True,'uniform_feasible_objective':uc,'optimal_no_worse_than_uniform':True,'zero_budget_detected_infeasible':True,'train_heldout_disjoint':True,'raw_source_sha256':json.loads((ROOT/'data/provenance.json').read_text())['raw_sha256']}
    split=pd.DataFrame(P['split']);assert not(set(split[split.split=='calibration'].sample_id)&set(split[split.split=='heldout'].sample_id))
    (out/'validation.json').write_text(json.dumps(checks,indent=2));pd.DataFrame([{k:v for k,v in r.items() if k!='checks'} for r in cases]).to_csv(out/'sensitivity_preview.csv',index=False)
    # Evaluation only: per-question hindsight quality oracle, without budget/cost constraints.
    df=pd.read_csv(ROOT/'data/routerbench_selected.csv')
    eval_rows=df.merge(split[['sample_id','split']],on='sample_id',validate='one_to_one')
    eval_rows=eval_rows[eval_rows['split']=='heldout']
    eval_scores=eval_rows[cfg['models']].to_numpy(float)
    oracle=float(eval_scores.max(axis=1).mean())
    extra={'heldout_quality_only_oracle_score':oracle,
           'heldout_fixed_model_scores':dict(zip(cfg['models'],eval_scores.mean(axis=0).tolist())),
           'heldout_routed_expected_score':result['heldout_expected_accuracy'],
           'quality_gap_to_oracle':oracle-result['heldout_expected_accuracy'],
           'oracle_note':'Hindsight per-question maximum score, ignoring budget/cost/switching. Quality upper bound only, not operational-objective regret.'}
    (out/'additional_checks.json').write_text(json.dumps(extra,indent=2))
    (out/'run_config.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='results');args=parser.parse_args();run(ROOT/args.out)
