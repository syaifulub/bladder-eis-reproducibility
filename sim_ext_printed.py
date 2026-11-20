SE_PRINTED = [
    ("sim ext: max naive optimism across departures", .422, lambda d: max(d[k]["naive_opt_max"] for k in ("correlated", "heteroscedastic", "heterogeneous")), 3),
    ("sim ext: min size-matched optimism across departures", -.027, lambda d: min(d[k]["matched_opt_min"] for k in ("correlated", "heteroscedastic", "heterogeneous")), 3),
    ("sim ext: max size-matched optimism across departures", .029, lambda d: max(d[k]["matched_opt_max"] for k in ("correlated", "heteroscedastic", "heterogeneous")), 3),
    ("sim ext: base reproduces naive null estimate", .935, lambda d: d["base"]["naive_est_null_icc097"], 3),
]
