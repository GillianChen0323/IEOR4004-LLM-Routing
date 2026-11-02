"""Optional full-source reconstruction. Downloads ~95 MB; run only when needed.
The source hash is pinned; compact scores/costs are written without raw text.
"""
from pathlib import Path
import hashlib,json,pickle,tempfile,urllib.request
import pandas as pd
ROOT=Path(__file__).resolve().parent
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={'pandas.core.frame':{'DataFrame'},'pandas.core.internals.managers':{'BlockManager'},'pandas._libs.internals':{'_unpickle_block'},'numpy.core.multiarray':{'_reconstruct'},'numpy':{'ndarray','dtype'},'builtins':{'slice'},'pandas.core.indexes.base':{'_new_Index','Index'},'pandas.core.indexes.range':{'RangeIndex'},'numpy.core.numeric':{'_frombuffer'}}
        if name not in allowed.get(module,set()):
            raise ValueError(f'Unexpected pickle global: {module}.{name}')
        return super().find_class(module,name)
def rebuild():
    meta=json.loads((ROOT/'data/provenance.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        raw=Path(tmp)/'routerbench_0shot.pkl'
        with urllib.request.urlopen(meta['source'],timeout=120) as response,raw.open('wb') as f:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                f.write(chunk)
        sha=hashlib.sha256(raw.read_bytes()).hexdigest()
        if sha!=meta['raw_sha256']:raise ValueError('Source hash changed: review before loading.')
        with raw.open('rb') as f:df=RestrictedUnpickler(f).load()
    mapping=meta['category_mapping'];models=meta['models']
    rows=df[df.eval_name.isin(mapping)].drop_duplicates('prompt').copy()
    rows['category']=rows.eval_name.map(mapping)
    cols=['sample_id','eval_name','category']+models+[m+'|total_cost' for m in models]
    assert len(rows)==meta['selected_rows']
    target=ROOT/'data/routerbench_selected.csv'
    # Verify exact values against existing data before replacing any file.
    rebuilt=rows[cols].reset_index(drop=True)
    if target.exists():
        old=pd.read_csv(target)
        pd.testing.assert_frame_equal(old,rebuilt,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-15)
    rebuilt.to_csv(target,index=False)
    print(f'Verified {len(rebuilt)} rows against pinned source {sha}')
if __name__=='__main__':rebuild()
