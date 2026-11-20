"""
semisynthetic_noneeg.py -- known-effect test on real units (PhysioNet Non-EEG, relaxation stages only).

The 80 relaxation stages of the 20 subjects are real units: their windows keep the real repeat noise, and their
subject-centred means keep the real between-unit variation. In each replicate, the units are shuffled into
synthetic subjects of n_h + 3 units: n_h observed reference-class units, one observed target unit, and two fresh
units (one of each class) that are never used for training or as a reference. A known class effect, delta times
the between-unit standard deviation of every feature, is added to the target units.
  estimate: leave-one-subject-out unit-level balanced accuracy on the observed units under the design;
  truth:    the same trained model applied to the fresh units, referenced against all n_h observed reference
            units (independent of the queries), as at deployment.
usage: python semisynthetic_noneeg.py n_h [reps]     outputs: results/noneeg/semisynthetic_nh{n_h}.csv
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = os.path.join("results", "noneeg")
FEATS = ["EDA_mean", "EDA_std", "EDA_slope", "TEMP_mean", "TEMP_slope", "ACC_std", "HR_mean", "HR_std", "SpO2_mean"]
DELTAS = [0.0, 0.5, 1.0, 2.0]


def load_blocks():
    W = pd.read_csv(os.path.join(OUT, "noneeg_windows.csv"))
    W = W[W.cond.str.startswith("relax")].copy()
    W[FEATS] = W[FEATS] - W.groupby("subject")[FEATS].transform("mean")          # remove subject baselines
    W[FEATS] = W[FEATS] / W[FEATS].std()                                          # common scale
    blocks = [g[FEATS].values for _, g in W.groupby("unit")]
    um = np.array([b.mean(0) for b in blocks]); ns = np.array([len(b) for b in blocks])
    sw2 = np.sum([((b - b.mean(0)) ** 2).sum(0) for b in blocks], 0) / (ns.sum() - len(blocks))
    sb = np.sqrt(np.clip(um.var(0, ddof=1) - sw2 * np.mean(1 / ns), 1e-6, None))
    return blocks, sb


def ref_feature(q, R):
    return np.abs(q - np.median(R, axis=0))


def build(subjects, design):
    """subjects: list of dicts with 'ref' (list of arrays) and 'tgt' (array). returns Z, y, subj, unit"""
    Z, y, S, U = [], [], [], []
    for s, sub in enumerate(subjects):
        R = sub["ref"]; nh = len(R); allR = np.vstack(R)
        for u, blk in enumerate(R):
            for j in range(len(blk)):
                if design == "LSO":
                    ref = np.delete(allR, sum(len(b) for b in R[:u]) + j, axis=0)
                else:
                    ref = np.vstack([R[v] for v in range(nh) if v != u])
                Z.append(ref_feature(blk[j], ref)); y.append(0); S.append(s); U.append(f"{s}r{u}")
        for j in range(len(sub["tgt"])):
            if design == "SM":
                for v in range(nh):
                    ref = np.vstack([R[w] for w in range(nh) if w != v])
                    Z.append(ref_feature(sub["tgt"][j], ref)); y.append(1); S.append(s); U.append(f"{s}t")
            else:
                Z.append(ref_feature(sub["tgt"][j], allR)); y.append(1); S.append(s); U.append(f"{s}t")
    return np.array(Z), np.array(y), np.array(S), np.array(U)


def unit_ba(units, y, p):
    d = pd.DataFrame({"u": units, "y": y, "p": p}).groupby("u").agg(y=("y", "first"), p=("p", "mean"))
    yh = (d.p.values >= 0.5).astype(int); yy = d.y.values
    return 0.5 * ((yh[yy == 1] == 1).mean() + (yh[yy == 0] == 0).mean())


def replicate(rng, blocks, sb, nh, delta, designs):
    order = rng.permutation(len(blocks)); k = nh + 3; nsub = len(blocks) // k; shift = delta * sb
    subjects = []
    for s in range(nsub):
        b = [blocks[i] for i in order[s * k:(s + 1) * k]]
        subjects.append(dict(ref=b[:nh], tgt=b[nh] + shift, fresh_ref=b[nh + 1], fresh_tgt=b[nh + 2] + shift))
    out = {}
    for design in designs:
        Z, y, S, U = build(subjects, design)
        est_p = np.zeros(len(y)); tr_u, tr_y, tr_p = [], [], []
        for s in range(nsub):
            te = S == s; tr = ~te
            w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
            sc = StandardScaler().fit(Z[tr]); m = LogisticRegression(C=1.0, max_iter=2000)
            m.fit(sc.transform(Z[tr]), y[tr], sample_weight=w)
            est_p[te] = m.predict_proba(sc.transform(Z[te]))[:, 1]
            allR = np.vstack(subjects[s]["ref"])                             # deployment: independent reference
            for lab, blk in [(0, subjects[s]["fresh_ref"]), (1, subjects[s]["fresh_tgt"])]:
                F = np.array([ref_feature(q, allR) for q in blk])
                tr_p += list(m.predict_proba(sc.transform(F))[:, 1]); tr_y += [lab] * len(blk); tr_u += [f"{s}f{lab}"] * len(blk)
        out[design] = (unit_ba(U, y, est_p), unit_ba(np.array(tr_u), np.array(tr_y), np.array(tr_p)))
    return out


def main():
    nh = int(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = np.random.default_rng(20261120 + nh)
    blocks, sb = load_blocks()
    designs = ["LSO"] if nh == 1 else ["LSO", "UW", "SM"]
    rows = []; t0 = time.time()
    for delta in DELTAS:
        for r in range(reps):
            res = replicate(rng, blocks, sb, nh, delta, designs)
            for d, (e, t) in res.items():
                rows.append(dict(n_h=nh, delta=delta, rep=r, design=d, estimate=e, truth=t, optimism=e - t))
        print(nh, delta, f"{time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, f"semisynthetic_nh{nh}.csv"), index=False)
    print(R.groupby(["design", "delta"])[["estimate", "truth", "optimism"]].mean().round(3))


if __name__ == "__main__":
    main()
