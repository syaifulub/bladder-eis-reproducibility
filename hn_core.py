"""
hn_core.py -- replication of reference-induced leakage on the public Electric_Bioimpedance_H&N_Dataset
(Missale 2022, Mendeley Data, https://doi.org/10.17632/b9t274zxgz.2, CC BY 4.0).
Place Dataset_EBI_3.csv in the repository root (or set HN_DATA). Outputs go to outputs_hn/.
Unit = patient x tissue type x status cell (the dataset has no specimen or puncture identifier).
"""
import os
DATA=os.environ.get("HN_DATA","Dataset_EBI_3.csv")
OUT="outputs_hn"; os.makedirs(OUT,exist_ok=True)
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X=[f'X{i}' for i in range(10,101,10)]; P=[f'P{i}' for i in range(10,101,10)]; F=X+P
def load(path):
    d=pd.read_csv(path,sep=';',decimal=',',encoding='utf-8-sig')
    d['y']=(d.Status=='Pathologic Tissue').astype(int)
    d['pt']=d.patientID.astype(str)+'|'+d.Type
    d['unit']=d.pt+'|'+d.Status
    return d.reset_index(drop=True)

def pairs(d):
    ok=d.groupby('pt').y.nunique(); ok=ok[ok==2].index
    return d[d.pt.isin(ok)].reset_index(drop=True)

def referenced(d, y):
    """leave-self-out patient-type reference: |x - median(healthy spectra of same patient-type, excluding query)|"""
    Z=np.zeros((len(d),len(F))); V=d[F].values
    for pt,idx in d.groupby('pt').indices.items():
        h=idx[y[idx]==0]
        for i in idx:
            r=h[h!=i]
            Z[i]=np.abs(V[i]-np.median(V[r],axis=0))
    return Z

def raw(d):
    T=pd.get_dummies(d.Type).values.astype(float)
    return np.c_[d[F].values,T]

def model(kind):
    if kind=='hgb':
        return HistGradientBoostingClassifier(max_iter=180,max_depth=3,learning_rate=0.04,min_samples_leaf=5,l2_regularization=2.0,random_state=0)
    return make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))

def ba(y,p):
    yhat=(p>=0.5).astype(int)
    se=(yhat[y==1]==1).mean(); sp=(yhat[y==0]==0).mean()
    return 0.5*(se+sp)

def lopo(d, Z, y, kind):
    """leave-one-patient-out; returns unit-level BA and spectrum-level BA"""
    pid=d.patientID.values; prob=np.zeros(len(d))
    w_cls={0:0.5/ (y==0).mean(),1:0.5/(y==1).mean()}
    for p in np.unique(pid):
        te=pid==p; tr=~te
        if len(np.unique(y[tr]))<2: prob[te]=0.5; continue
        m=model(kind); sw=np.array([w_cls[v] for v in y[tr]])
        if kind=='hgb': m.fit(Z[tr],y[tr],sample_weight=sw)
        else: m.fit(Z[tr],y[tr],logisticregression__sample_weight=sw)
        prob[te]=m.predict_proba(Z[te])[:,1]
    u=pd.DataFrame(dict(unit=d.unit,y=y,p=prob)).groupby('unit').agg(y=('y','first'),p=('p','mean'))
    return ba(u.y.values,u.p.values), ba(y,prob)

def icc1(groups, x):
    """one-way random-effects ICC(1) with unequal group sizes"""
    df=pd.DataFrame(dict(g=groups,x=x)); k=df.g.nunique(); N=len(df)
    gm=df.groupby('g').x.agg(['mean','size']); grand=df.x.mean()
    ssb=(gm['size']*(gm['mean']-grand)**2).sum(); ssw=((df.x-df.g.map(gm['mean']))**2).sum()
    msb=ssb/(k-1); msw=ssw/(N-k); n0=(N-(gm['size']**2).sum()/N)/(k-1)
    return (msb-msw)/(msb+(n0-1)*msw), msb, msw, n0
