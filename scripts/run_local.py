"""Generate fixtures if necessary and evaluate SYNTHETIC ONLY results."""
import argparse
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import run_pipeline
from src.scoring import pnl, validate_submission
from src.synthetic import generate_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/synthetic')
    parser.add_argument('--no-lightgbm', action='store_true')
    parser.add_argument('--wf-days', type=int, default=7)
    parser.add_argument('--archive-days', type=int, default=30)
    args = parser.parse_args()
    start = time.perf_counter()
    root = Path(args.data_dir)
    if not (root/'test.csv').exists():
        generate_synthetic(root)
    print('SYNTHETIC ONLY — not competition performance')
    submission, table = run_pipeline(root, wf_days=args.wf_days, archive_days=args.archive_days,
                                     use_lightgbm=not args.no_lightgbm)
    validate_submission(submission, pd.read_csv(root/'test.csv'))
    print('SYNTHETIC ONLY walk-forward (in-sample-tuned on training OOF):')
    print(table.to_string(index=False))
    if len(table):
        print(f'OOF totals: PnL={table.pnl.sum():.6f}; n buys={table.n_buys.sum()}; hits={table.n_hits.sum()}; default PnL={table.default_pnl.sum():.6f}; baseline tp=5 PnL={table.baseline_pnl.sum():.6f}')
    answers = pd.read_csv(root.parent/(root.name+'_answers')/'test_labels.csv')
    scored = submission.merge(answers, on='mint', validate='one_to_one')
    if len(scored) != len(submission):
        raise ValueError('Hidden labels do not match test mints')
    gains = pnl(scored.tp, scored.max_x, scored.final_x)
    buys = scored.tp.gt(0)
    print(f'SYNTHETIC ONLY test: PnL={gains.sum():.6f}; n buys={buys.sum()}; hits={(buys & scored.max_x.ge(scored.tp)).sum()}')
    print(f'Total runtime: {time.perf_counter()-start:.3f}s')


if __name__ == '__main__':
    main()
