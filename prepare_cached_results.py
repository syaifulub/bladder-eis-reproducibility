"""Stage bundled results/ as outputs_ijs/ without touching the reference files.

Usage:
    python prepare_cached_results.py
    python prepare_cached_results.py --overwrite
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='replace existing output files')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    source = base / 'results'
    target = base / 'outputs_ijs'
    if not source.is_dir():
        parser.error(f'Bundled results folder not found: {source}')
    new, skipped = 0, 0
    for src in sorted(source.rglob('*')):
        if not src.is_file():
            continue
        dst = target / src.relative_to(source)
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        new += 1
    print(f'Staged {new} file(s) to {target.name}/; skipped {skipped} existing file(s).')
    print('Reference results/ was not modified.')


if __name__ == '__main__':
    main()
