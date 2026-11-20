"""
noneeg_io.py -- minimal WFDB reader (format 16 signals, MIT-format annotations) for the PhysioNet
Non-EEG Dataset for Assessment of Neurological Status v1.0.0 (Birjandtalab et al., SiPS 2016;
https://physionet.org/content/noneeg/1.0.0/, ODC-By 1.0). No dependency on the wfdb package.
"""
import os, re
import numpy as np


def read_header(path_hea):
    lines = [l.strip() for l in open(path_hea) if l.strip() and not l.startswith("#")]
    name, nsig, fs, nsamp = lines[0].split()[:4]
    sigs = []
    for l in lines[1:1 + int(nsig)]:
        p = l.split()
        m = re.match(r"([-\d.eE+]+)(?:\(([-\d]+)\))?(?:/(\S+))?", p[2])
        gain = float(m.group(1)); base = int(m.group(2)) if m.group(2) else int(p[4]) if len(p) > 4 else 0
        sigs.append(dict(file=p[0], fmt=p[1], gain=gain, baseline=base, units=m.group(3) or "", name=p[-1]))
    return dict(name=name, nsig=int(nsig), fs=float(fs), nsamp=int(nsamp), signals=sigs)


def read_record(directory, record):
    h = read_header(os.path.join(directory, record + ".hea"))
    assert all(s["fmt"] == "16" for s in h["signals"]), "only format 16 supported"
    raw = np.fromfile(os.path.join(directory, h["signals"][0]["file"]), dtype="<i2")
    raw = raw[: (len(raw) // h["nsig"]) * h["nsig"]].reshape(-1, h["nsig"])
    phys = {s["name"]: (raw[:, k].astype(float) - s["baseline"]) / s["gain"] for k, s in enumerate(h["signals"])}
    return h, phys


def read_annotations(path_atr):
    """MIT annotation format: returns list of (sample, code, aux)."""
    b = np.fromfile(path_atr, dtype="<u2")
    out, t, i = [], 0, 0
    while i < len(b):
        w = int(b[i]); code = w >> 10; val = w & 0x3FF
        if code == 0 and val == 0:
            break
        if code == 59:                                   # SKIP: 32-bit interval in the next two words
            hi, lo = int(b[i + 1]), int(b[i + 2])
            t += (hi << 16) | lo; i += 3; continue
        if code == 63:                                   # AUX: string of length val attached to previous annotation
            nbytes = val; nwords = (nbytes + 1) // 2
            s = b[i + 1:i + 1 + nwords].tobytes()[:nbytes].decode("latin1").rstrip("\x00")
            if out:
                out[-1] = (out[-1][0], out[-1][1], s)
            i += 1 + nwords; continue
        if code in (60, 61, 62):                         # NUM, SUB, CHN: ignore
            i += 1; continue
        t += val; out.append((t, code, "")); i += 1
    return out
