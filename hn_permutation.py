"""Usage: python hn_permutation.py {lr|hgb} B [seed] [tag]  -- within-pair label permutation null with references rebuilt."""
import os, sys, json, time; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, hn_core as c
kind=sys.argv[1]; B=int(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 20261119; tag=sys.argv[4] if len(sys.argv)>4 else ''
d=c.load(c.DATA); dp=c.pairs(d); y0=dp.y.values
rng=np.random.default_rng(seed)
R=c.raw(dp)
obs=dict(sw=c.lopo(dp,c.referenced(dp,y0),y0,kind), raw=c.lopo(dp,R,y0,kind))
null=dict(sw=[],raw=[])
pts=dp.pt.unique()
for b in range(B):
    flip=dict(zip(pts,rng.random(len(pts))<0.5))
    y=np.where(dp.pt.map(flip).values,1-y0,y0)
    null['sw'].append(c.lopo(dp,c.referenced(dp,y),y,kind))
    null['raw'].append(c.lopo(dp,R,y,kind))
out=dict(kind=kind,B=B,obs=obs,null={k:np.array(v).tolist() for k,v in null.items()})
json.dump(out,open(f'{c.OUT}/perm_{kind}{tag}.json','w'))
for k in null:
    a=np.array(null[k])[:,0]
    print(kind,k,'obs unit BA',round(obs[k][0],3),'null mean',round(a.mean(),3),'95%',np.round(np.percentile(a,[2.5,97.5]),3),'p',round((1+(a>=obs[k][0]).sum())/(1+B),4))
