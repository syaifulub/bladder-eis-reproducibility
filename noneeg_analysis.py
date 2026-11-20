"""
noneeg_analysis.py -- reference-induced leakage on the PhysioNet Non-EEG dataset (20 subjects, wrist sensors).

Units: the protocol stages of each subject. Reference class: the four relaxation stages; target class: physical,
cognitive and (main) emotional stress. The 40-s instruction segment before the cognitive task is excluded.
Repeats: non-overlapping 30-s windows. Designs: leave-self-out (LSO), unit-wise (UW), size-matched (SM), unreferenced.

usage: python noneeg_analysis.py DATA_DIR [config] [kind] [n_perm] [seed] [tag]
  config: main | noacc | nophysical      kind: lr | hgb | none (observed + ICC + granularity + index only)
Outputs: results/noneeg/
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noneeg_io as nio

OUT = os.path.join("results", "noneeg"); os.makedirs(OUT, exist_ok=True)
FS, WIN = 8, 30
STAGE_NAMES = ["relax1", "physical", "relax2", "instruction", "cognitive", "relax3", "emotional", "relax4"]
ALL_FEATS = ["EDA_mean", "EDA_std", "EDA_slope", "TEMP_mean", "TEMP_slope", "ACC_std", "HR_mean", "HR_std", "SpO2_mean"]


def slope_per_min(v, fs):
    t = np.arange(len(v)) / fs / 60.0
    return np.polyfit(t, v, 1)[0]


def build_windows(D):
    rows = []
    for s in range(1, 21):
        h, p = nio.read_record(D, f"Subject{s}_AccTempEDA")
        h2, q = nio.read_record(D, f"Subject{s}_SpO2HR")
        ann = nio.read_annotations(os.path.join(D, f"Subject{s}_AccTempEDA.atr"))
        starts = [a[0] for a in ann] + [h["nsamp"]]
        labels = [a[2] for a in ann]
        assert len(labels) == 8, (s, labels)
        mag = np.sqrt(p["ax"] ** 2 + p["ay"] ** 2 + p["az"] ** 2)
        for k, name in enumerate(STAGE_NAMES):
            if name == "instruction":
                continue
            a0, a1 = starts[k], starts[k + 1]
            for j, w0 in enumerate(range(a0, a1 - WIN * FS + 1, WIN * FS)):
                w1 = w0 + WIN * FS; b0, b1 = w0 // FS, w1 // FS
                eda, tmp = p["EDA"][w0:w1], p["temp"][w0:w1]
                hr, sp = q["hr"][b0:b1], q["SpO2"][b0:b1]
                rows.append([f"S{s:02d}", name, j, eda.mean(), eda.std(), slope_per_min(eda, FS), tmp.mean(),
                             slope_per_min(tmp, FS), mag[w0:w1].std(), hr.mean(), hr.std(), sp.mean()])
    W = pd.DataFrame(rows, columns=["subject", "cond", "win"] + ALL_FEATS)
    W["y"] = W.cond.isin(["physical", "cognitive", "emotional"]).astype(int)
    W["unit"] = W.subject + "|" + W.cond
    return W


class Analysis:
    def __init__(self, W, feats, seed=20261119):
        self.W = W.reset_index(drop=True); self.F = feats
        self.X = self.W[feats].values.astype(float)
        self.SUBJ = self.W.subject.values; self.UNIT = self.W.unit.values; self.Y = self.W.y.values
        self.SUBJECTS = np.unique(self.SUBJ)
        self.SI = {s: np.flatnonzero(self.SUBJ == s) for s in self.SUBJECTS}
        self.UNITS_OF = {s: np.unique(self.UNIT[i]) for s, i in self.SI.items()}
        self.NT = {s: int(len(np.unique(self.UNIT[i][self.Y[i] == 1]))) for s, i in self.SI.items()}
        self.rng = np.random.default_rng(seed)

    def referenced(self, y, design, Xs=None, stat="median", absval=True):
        X = self.X if Xs is None else Xs
        f = (lambda A: np.median(A, axis=0)) if stat == "median" else (lambda A: A.mean(axis=0))
        Z, src = [], []
        for s, idx in self.SI.items():
            ref = idx[y[idx] == 0]; ru = np.unique(self.UNIT[ref])
            excl = {u: f(X[ref[self.UNIT[ref] != u]]) for u in ru} if design in ("UW", "SM") else None
            full = f(X[ref])
            for i in idx:
                if y[i] == 0:
                    c = f(X[ref[ref != i]]) if design == "LSO" else excl[self.UNIT[i]]
                    Z.append(X[i] - c); src.append(i)
                elif design == "SM":
                    for u in ru:
                        Z.append(X[i] - excl[u]); src.append(i)
                else:
                    Z.append(X[i] - full); src.append(i)
        Z = np.asarray(Z)
        return (np.abs(Z) if absval else Z), np.asarray(src)

    @staticmethod
    def weights(subj, y):
        w = np.zeros(len(y))
        for s in np.unique(subj):
            for c in (0, 1):
                m = (subj == s) & (y == c); k = m.sum()
                if k: w[m] = 1.0 / k
        return w * len(y) / w.sum()

    @staticmethod
    def model(kind, seed=0):
        if kind == "lr":
            return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        return HistGradientBoostingClassifier(max_iter=180, max_depth=3, learning_rate=0.04, min_samples_leaf=5,
                                              l2_regularization=2.0, random_state=seed)

    @staticmethod
    def ba(y, p):
        yh = (p >= 0.5).astype(int)
        return 0.5 * ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean())

    def fit(self, m, kind, Z, y, w):
        if kind == "lr": m.fit(Z, y, logisticregression__sample_weight=w)
        else: m.fit(Z, y, sample_weight=w)

    def loso(self, Z, src, y, kind):
        ys, ss = y[src], self.SUBJ[src]; prob = np.zeros(len(src))
        for s in self.SUBJECTS:
            te = ss == s; tr = ~te
            m = self.model(kind); self.fit(m, kind, Z[tr], ys[tr], self.weights(ss[tr], ys[tr]))
            prob[te] = m.predict_proba(Z[te])[:, 1]
        pw = pd.Series(prob).groupby(src).mean()
        win = pd.DataFrame({"p": pw.values, "y": y[pw.index.values], "unit": self.UNIT[pw.index.values]})
        un = win.groupby("unit").agg(p=("p", "mean"), y=("y", "first"))
        return self.ba(un.y.values, un.p.values), self.ba(win.y.values, win.p.values)

    def evaluate(self, y, kind):
        out = {d: self.loso(*self.referenced(y, d), y, kind)[0] for d in ["LSO", "UW", "SM"]}
        out["UNREF"] = self.loso(self.X, np.arange(len(self.W)), y, kind)[0]
        return out

    def permuted(self):
        y = np.zeros(len(self.W), int)
        for s, idx in self.SI.items():
            us = self.rng.choice(self.UNITS_OF[s], size=self.NT[s], replace=False)
            y[idx[np.isin(self.UNIT[idx], us)]] = 1
        return y

    # ---------------------------------------------------------------- descriptive pieces
    def icc(self):
        R = self.W.copy()
        for f in self.F:
            R[f] = R[f] - R.groupby("subject")[f].transform("mean")
            R[f] = R[f] - R.groupby("cond")[f].transform("mean")
        size = self.W.groupby("unit").size(); mbar = size.mean(); cv = size.std(ddof=0) / mbar
        rows = []
        for f in self.F:
            df = pd.DataFrame({"g": R.unit.values, "x": R[f].values}); k = df.g.nunique(); N = len(df)
            gm = df.groupby("g").x.agg(["mean", "size"]); grand = df.x.mean()
            msb = (gm["size"] * (gm["mean"] - grand) ** 2).sum() / (k - 1)
            msw = ((df.x - df.g.map(gm["mean"])) ** 2).sum() / (N - k)
            n0 = (N - (gm["size"] ** 2).sum() / N) / (k - 1)
            icc = (msb - msw) / (msb + (n0 - 1) * msw)
            de = 1 + ((1 + cv ** 2) * mbar - 1) * icc
            rows.append(dict(feature=f, ICC=icc, DE=de, N_eff=N / de))
        return pd.DataFrame(rows), float(mbar)

    def granularity(self, kind, reps=5):
        X, Y, S, U = self.X, self.Y, self.SUBJ, self.UNIT
        def run(splits, seed):
            prob = np.zeros(len(Y))
            for tr, te in splits:
                m = self.model(kind, seed); self.fit(m, kind, X[tr], Y[tr], self.weights(S[tr], Y[tr]))
                prob[te] = m.predict_proba(X[te])[:, 1]
            return self.ba(Y, prob)
        return {
            "random_windows_5fold": float(np.mean([run(StratifiedKFold(5, shuffle=True, random_state=r).split(X, Y), r) for r in range(reps)])),
            "grouped_by_unit_5fold": float(np.mean([run(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, Y, U), r) for r in range(reps)])),
            "leave_one_subject_out": float(run([(np.flatnonzero(S != s), np.flatnonzero(S == s)) for s in self.SUBJECTS], 0)),
        }

    def index(self, n_perm):
        Xs = self.X.copy()
        for s, idx in self.SI.items():
            Xs[idx] = (self.X[idx] - self.X[idx].mean(0)) / (self.X[idx].std(0) + 1e-12)
        df = pd.DataFrame(Xs, columns=self.F); df["unit"] = self.UNIT; df["subject"] = self.SUBJ
        um = df.groupby("unit")[self.F].transform("mean")
        sw2 = ((df[self.F] - um) ** 2).sum().values / (len(df) - df.unit.nunique())
        g = df.groupby(["subject", "unit"])[self.F].mean(); ns = df.groupby(["subject", "unit"]).size()
        sb2 = np.clip(np.mean([g.loc[s].var(ddof=1).values - sw2 * (1.0 / ns.loc[s]).mean() for s in self.SUBJECTS], 0), 1e-6, None)
        n_h = float(np.mean([len(self.UNITS_OF[s]) - self.NT[s] for s in self.SUBJECTS])); m = df.groupby("unit").size().mean()
        def V(d):
            n = n_h * m - 1; vt = sb2 * (1 + 1 / n_h) + sw2 * (1 + 1 / (n_h * m))
            if d == "LSO":
                vr = sb2 * m * m * n_h * (n_h - 1) / n ** 2 + sw2 * (1 + 1 / n)
            else:
                vr = sb2 * (1 + 1 / (n_h - 1)) + sw2 * (1 + 1 / ((n_h - 1) * m))
                if d == "SM": vt = vr
            return vr, vt
        res = {"n_h": n_h, "m_bar": float(m), "ICC_within_subject": dict(zip(self.F, (sb2 / (sb2 + sw2)).round(4).tolist()))}
        for d in ["LSO", "UW", "SM"]:
            vr, vt = V(d); rho = float(np.exp(np.mean(np.log(vt / vr))))
            Zr, src = self.referenced(self.Y, d, Xs=Xs, stat="mean", absval=False)
            C = np.nan_to_num(np.corrcoef(Zr[self.Y[src] == 0].T)); np.fill_diagonal(C, 1)
            lam = np.clip(np.linalg.eigvalsh(C), 0, None); deff = float(lam.sum() ** 2 / (lam ** 2).sum())
            aucs = []
            for b in range(n_perm):
                y = self.permuted(); Z, src = self.referenced(y, d, Xs=Xs, stat="mean", absval=False)
                sc = (Z ** 2 / vr).sum(1); t, r = sc[y[src] == 1], sc[y[src] == 0]
                aucs.append(stats.mannwhitneyu(t, r).statistic / (len(t) * len(r)))
            res[d] = dict(rho=rho, d_eff=deff, predicted_null_AUC=float(stats.f.cdf(rho, deff, deff)),
                          observed_null_AUC=float(np.mean(aucs)), observed_95=np.percentile(aucs, [2.5, 97.5]).tolist())
        return res


def configure(W, config):
    feats = list(ALL_FEATS)
    if config == "noacc":
        feats.remove("ACC_std")
    if config == "nophysical":
        W = W[W.cond != "physical"]
    return W, feats


def main():
    D = sys.argv[1]; config = sys.argv[2] if len(sys.argv) > 2 else "main"
    kind = sys.argv[3] if len(sys.argv) > 3 else "none"; B = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20261119; tag = sys.argv[6] if len(sys.argv) > 6 else ""
    fcsv = os.path.join(OUT, "noneeg_windows.csv")
    if not os.path.exists(fcsv):
        build_windows(D).to_csv(fcsv, index=False)
    W, feats = configure(pd.read_csv(fcsv), config)
    A = Analysis(W, feats, seed=seed)
    t0 = time.time()
    if kind == "none":
        icc, mbar = A.icc()
        res = dict(config=config, n_subjects=len(A.SUBJECTS), n_units=int(W.unit.nunique()), n_windows=len(W),
                   mean_windows_per_unit=mbar, windows_per_cond=W.groupby("cond").size().to_dict(),
                   icc=icc.round(4).to_dict(orient="list"),
                   granularity={k: A.granularity(k) for k in ["lr", "hgb"]},
                   observed={k: A.evaluate(A.Y, k) for k in ["lr", "hgb"]},
                   index=A.index(200))
        json.dump(res, open(os.path.join(OUT, f"summary_{config}.json"), "w"), indent=1, default=float)
        print(json.dumps(res, indent=1, default=float))
    else:
        obs = A.evaluate(A.Y, kind)
        null = {d: [] for d in obs}
        for b in range(B):
            r = A.evaluate(A.permuted(), kind)
            for d in r: null[d].append(r[d])
            if (b + 1) % max(1, B // 10) == 0:
                print(f"{config} {kind}: {b+1}/{B} {time.time()-t0:.0f}s", flush=True)
        json.dump(dict(config=config, kind=kind, B=B, seed=seed, observed=obs, null=null),
                  open(os.path.join(OUT, f"perm_{config}_{kind}{tag}.json"), "w"))
        for d in obs:
            v = np.array(null[d]); print(config, kind, d, round(obs[d], 3), "null", round(v.mean(), 3),
                                         np.round(np.percentile(v, [2.5, 97.5]), 3), "p", round((1 + (v >= obs[d]).sum()) / (1 + B), 4))


if __name__ == "__main__":
    main()
