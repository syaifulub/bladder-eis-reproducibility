SEL_PRINTED = [
    ("selection-aware: permutations", 100, ["n_perm"], 0),
    ("selection-aware: observed", .950, ["BA_selected", "observed"], 3),
    ("selection-aware: null mean", .471, ["BA_selected", "null_mean"], 3),
    ("selection-aware: null 2.5%", .315, ["BA_selected", "null_q025"], 3),
    ("selection-aware: null 97.5%", .691, ["BA_selected", "null_q975"], 3),
    ("selection-aware: p", .0099, ["BA_selected", "p_value"], 4),
    ("selection-aware: observed above all permutations", 1, ["_above_all"], 0),
    ("pre-specified (same runs): observed", .819, ["BA_prespecified", "observed"], 3),
    ("pre-specified (same runs): null mean", .455, ["BA_prespecified", "null_mean"], 3),
    ("pre-specified (same runs): p", .0099, ["BA_prespecified", "p_value"], 4),
]
