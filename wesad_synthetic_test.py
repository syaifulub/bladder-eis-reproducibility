"""Synthetic data in the WESAD pickle format, for smoke tests of the notebook only.
usage: python make_synthetic.py OUTDIR [null]
'null': no stress effect, independent unit effect per segment, long segments, 8 subjects (leakage check)."""
import os, pickle, sys
import numpy as np

FS = 700
out = sys.argv[1]
NULL = len(sys.argv) > 2 and sys.argv[2] == "null"
rng = np.random.default_rng(1)
EFF = 0.0 if NULL else 1.0
if NULL:
    plan = [(0, 0.5), (1, 20), (0, 0.5), (3, 7), (0, 0.5), (4, 7), (0, 0.5), (2, 10), (0, 0.5), (4, 7), (0, 0.5)]
    subjects = [2, 3, 4, 5, 6, 7, 8, 9]
else:
    plan = [(0, 0.5), (1, 6), (0, 0.5), (3, 2), (0, 0.5), (4, 2), (0, 0.5), (2, 4), (0, 0.5), (4, 2), (0, 0.5)]
    subjects = [2, 3, 4, 5, 6]

for s in subjects:
    parts = {k: [] for k in ["ecg", "eda", "resp", "temp", "emg", "acc", "lab"]}
    base_hr, base_eda, base_temp = 70 + 5 * rng.normal(), 2 + rng.normal(), 33 + rng.normal()
    phase = 0.0; rphase = 0.0
    for c, mins in plan:
        n = int(mins * 60 * FS); t = np.arange(n) / FS
        u = rng.normal(0, 1)                                  # independent effect of this segment (unit)
        stress = EFF * (c == 2)
        hr = base_hr + 12 * stress + 3 * u + 0.5 * rng.standard_normal(n).cumsum() / np.sqrt(n)
        ph = phase + np.cumsum(hr / 60 / FS); phase = ph[-1]
        parts["ecg"].append(np.exp(-((ph % 1) - 0.5) ** 2 / 0.0005) + 0.05 * rng.standard_normal(n))
        parts["eda"].append(base_eda + 0.5 * stress + 0.3 * u + 0.01 * rng.standard_normal(n))
        rr = 0.25 + 0.05 * stress + 0.02 * u + 0.01 * np.sin(2 * np.pi * t / 97)
        rp = rphase + np.cumsum(rr) / FS; rphase = rp[-1]
        parts["resp"].append(np.sin(2 * np.pi * rp) + 0.1 * rng.standard_normal(n))
        parts["temp"].append(base_temp + 0.2 * u + 0.1 * t / 600 + 0.05 * rng.standard_normal(n))
        parts["emg"].append(rng.standard_normal(n) * (1 + 0.3 * stress + 0.1 * abs(u)))
        parts["acc"].append(rng.standard_normal((n, 3)) * 0.02 * (1 + 0.2 * abs(u)))
        parts["lab"].append(np.full(n, c, np.int32))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    chest = {"ACC": cat["acc"].astype(np.float32)}
    for k, key in [("ecg", "ECG"), ("emg", "EMG"), ("eda", "EDA"), ("temp", "Temp"), ("resp", "Resp")]:
        chest[key] = cat[k][:, None].astype(np.float32)
    d = {"signal": {"chest": chest, "wrist": {}}, "label": cat["lab"], "subject": f"S{s}"}
    os.makedirs(os.path.join(out, f"S{s}"), exist_ok=True)
    with open(os.path.join(out, f"S{s}", f"S{s}.pkl"), "wb") as f:
        pickle.dump(d, f)
    print(f"S{s}", len(cat["lab"]))
