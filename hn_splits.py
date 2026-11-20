"""Usage: python hn_splits.py {lr|hgb}  -- random-spectrum vs unit-grouped vs leave-one-patient-out validation."""
import os, sys, json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, hn_core as c
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
kind=sys.argv[1]
d=c.load(c.DATA); y=d.y.values; Z=c.raw(d)
def run(splits):
    prob=np.zeros(len(d))
    for tr,te in splits:
        w=np.where(y[tr]==1,0.5/y[tr].mean(),0.5/(1-y[tr].mean()))
        m=c.model(kind)
        if kind=='hgb': m.fit(Z[tr],y[tr],sample_weight=w)
        else: m.fit(Z[tr],y[tr],logisticregression__sample_weight=w)
        prob[te]=m.predict_proba(Z[te])[:,1]
    u=pd.DataFrame(dict(unit=d.unit,y=y,p=prob)).groupby('unit').agg(y=('y','first'),p=('p','mean'))
    return round(c.ba(y,prob),3), round(c.ba(u.y.values,u.p.values),3)
res={}
res['random spectra (5-fold)']=[run(StratifiedKFold(5,shuffle=True,random_state=s).split(Z,y)) for s in range(5)]
res['grouped by unit (5-fold)']=[run(StratifiedGroupKFold(5,shuffle=True,random_state=s).split(Z,y,d.unit)) for s in range(5)]
res['leave-one-patient-out']=[run(LeaveOneGroupOut().split(Z,y,d.patientID))]
out={k:dict(spectrum_BA=float(np.mean([v[0] for v in vs])),unit_BA=float(np.mean([v[1] for v in vs]))) for k,vs in res.items()}
print(kind, json.dumps(out,indent=1)); json.dump(out,open(f'{c.OUT}/splits_{kind}.json','w'),indent=1)
