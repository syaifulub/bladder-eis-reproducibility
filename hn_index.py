"""
hn_index.py -- reference-leakage index (Corollary 1) on the head-and-neck dataset, compared with the
empirical null AUC of the whitened leave-self-out deviation score over within-pair label permutations.
Index: AUC0 = F_{d,d}(rho), rho = V_t / V_r (Eq. 2 with n_h = 1), d = effective dimension.
"""
import os, sys, json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, hn_core as c
from scipy import stats
d=c.load(c.DATA); dp=c.pairs(d).copy(); F=c.F
for t,idx in dp.groupby('Type').groups.items():                      # standardize within tissue type
    dp.loc[idx,F]=(dp.loc[idx,F]-dp.loc[idx,F].mean())/dp.loc[idx,F].std()
sw2=[]; sd2=[]
for f in F:                                                           # variance components within pairs
    g=dp.groupby('unit')[f]; w=((dp[f]-g.transform('mean'))**2).sum()/(len(dp)-dp.unit.nunique()); sw2.append(w)
    um=g.mean(); us=g.size(); diffs=[]
    for pt,u in dp.groupby('pt').unit.unique().items():
        a,b=u; diffs.append((um[a]-um[b])**2-w*(1/us[a]+1/us[b]))     # estimates E(theta_A-theta_B)^2 = 2 sigma_b^2
    sd2.append(np.mean(diffs))
sw2=np.array(sw2); sd2=np.array(sd2); mh=dp.groupby('unit').size().mean()
Vr=sw2*(1+1/(mh-1)); Vt=sd2+sw2*(1+1/mh); rho=Vt/Vr
V=dp[F].values
def devs(y):
    D=np.zeros((len(dp),len(F)))
    for pt,idx in dp.groupby('pt').indices.items():
        h=idx[y[idx]==0]
        for i in idx: r=h[h!=i]; D[i]=V[i]-V[r].mean(0)
    return D
y0=dp.y.values; D=devs(y0)
lam=np.linalg.eigvalsh(np.corrcoef(D[y0==0].T)); deff=lam.sum()**2/(lam**2).sum()
rbar=float(np.exp(np.mean(np.log(rho))))
pred=float(stats.f.cdf(rbar,deff,deff))
rng=np.random.default_rng(5); pts=dp.pt.unique(); obs=[]
for b in range(200):
    flip=dict(zip(pts,rng.random(len(pts))<0.5)); y=np.where(dp.pt.map(flip).values,1-y0,y0)
    S=((devs(y)**2)/Vr).sum(1); st,sh=S[y==1],S[y==0]
    obs.append(stats.mannwhitneyu(st,sh).statistic/(len(st)*len(sh)))
out=dict(rho_per_feature=dict(zip(F,np.round(rho,4).tolist())),rho_geomean=rbar,d_eff=float(deff),mean_repeats=float(mh),
         predicted_null_AUC=pred,predicted_d1=float(stats.f.cdf(rbar,1,1)),observed_null_AUC_mean=float(np.mean(obs)),
         observed_null_AUC_95=np.percentile(obs,[2.5,97.5]).tolist(),n_perm=200)
json.dump(out,open(f'{c.OUT}/hn_index.json','w'),indent=1); print(json.dumps({k:v for k,v in out.items() if k!='rho_per_feature'},indent=1))
