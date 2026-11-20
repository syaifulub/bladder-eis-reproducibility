"""ICC(1) per feature after removing tissue type x status means; Kish design effect and effective N."""
import os, sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, hn_core as c
d=c.load(c.DATA)
rows=[]
for f in c.F:
    r=d[f]-d.groupby(['Type','Status'])[f].transform('mean')
    rows.append(dict(feature=f,ICC=c.icc1(d.unit.values,r.values)[0]))
T=pd.DataFrame(rows)
size=d.groupby('unit').size(); mbar=size.mean(); cv=size.std(ddof=0)/mbar
T['DE']=1+((1+cv**2)*mbar-1)*T.ICC; T['N_eff']=len(d)/T.DE
T.to_csv(f'{c.OUT}/hn_icc.csv',index=False); print(T.round(3).to_string()); print('units',d.unit.nunique(),'mean repeats',round(mbar,2))
