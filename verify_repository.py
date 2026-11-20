"""Fast static check of repository content (not a fresh scientific rerun)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    problems = []
    required = (
        'README.md', 'LICENSE', 'CITATION.cff', 'requirements.txt',
        'Bladder_EIS_IJS_Full_Reproduction.ipynb',
        'eis_pipeline.py', 'eis_groups.py', 'cole_analysis.py',
        'ijs_analysis.py', 'ijs_revision.py', 'ijs_selection_permutation.py',
        'article_tables.py', 'prepare_cached_results.py',
        'results/consistency_check.csv',
        'results/revision/selection_permutation_results.json',
    )
    for file in required:
        if not (root / file).is_file():
            problems.append(f'Missing file: {file}')
    for stem in ('Fig1_Referencing_Designs', 'Fig2_Sample_Structure',
                 'Fig3_Selected_Frequencies', 'Fig4_Permutation_Null', 'Fig5_Simulation'):
        if not (root / 'figs' / f'{stem}.png').is_file():
            problems.append(f'Missing published figure: {stem}.png')
    py_paths = list(root.glob('*.py'))
    for path in py_paths:
        try:
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        except Exception as exc:
            problems.append(f'Python syntax issue in {path.name}: {exc}')
    json_paths = list((root / 'results').rglob('*.json'))
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            problems.append(f'Invalid JSON in {path.relative_to(root)}: {exc}')
    try:
        nb = json.loads((root / 'Bladder_EIS_IJS_Full_Reproduction.ipynb').read_text(encoding='utf-8'))
        count = len(nb['cells'])
        if count < 75:
            problems.append('Notebook unexpectedly short (expected the supplied expanded version)')
    except Exception as exc:
        count = 0
        problems.append(f'Invalid notebook JSON: {exc}')
    try:
        with (root / 'results/consistency_check.csv').open(encoding='utf-8', newline='') as f:
            entries = list(csv.DictReader(f))
        passed = sum(x['Status'] == 'PASS' for x in entries)
        if passed != len(entries) or len(entries) != 464:
            problems.append(f'Unexpected saved consistency status: {passed}/{len(entries)} PASS (expected 464/464)')
    except Exception as exc:
        passed, entries = 0, []
        problems.append(f'Cannot check saved consistency CSV: {exc}')
    try:
        per = json.loads((root/'results/revision/selection_permutation_results.json').read_text(encoding='utf-8'))
        num_perm = per['n_perm']
        if num_perm != 100:
            problems.append(f'Unexpected supplied selection-aware n_perm={num_perm}, expected 100')
    except Exception as exc:
        num_perm = '?'
        problems.append(f'Cannot verify supplied selection-aware permutation result: {exc}')
    print(f'Python source files: {len(py_paths)}')
    print(f'Saved result JSON files: {len(json_paths)}')
    print(f'Notebook cells: {count}')
    print(f'Saved consistency-check entries: {passed}/{len(entries)} PASS')
    print(f'Saved selection-aware permutations: {num_perm}')
    if problems:
        print('PACKAGE CHECK FAILED:')
        for msg in problems:
            print(' - ' + msg)
        return 1
    print('PACKAGE CHECK PASSED (static/content check only; experiments were not rerun).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
