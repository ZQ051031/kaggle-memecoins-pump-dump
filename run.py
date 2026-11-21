"""Run the offline competition pipeline."""
import argparse

from src.model import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--no-lightgbm', action='store_true')
    parser.add_argument('--wf-days', type=int, default=7)
    parser.add_argument('--archive-days', type=int, default=30)
    args = parser.parse_args()
    _, table = run_pipeline(args.data_dir, [args.out, '/kaggle/working/submission.csv'],
                            args.wf_days, args.archive_days, not args.no_lightgbm)
    print('Walk-forward (in-sample-tuned on training OOF):')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
