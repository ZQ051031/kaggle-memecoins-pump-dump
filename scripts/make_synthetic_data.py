"""Generate small, deterministic competition-shaped fixtures."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synthetic import generate_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/synthetic')
    parser.add_argument('--days', type=int, default=40)
    parser.add_argument('--tokens-per-day', type=int, default=80)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    labels = generate_synthetic(args.out, args.days, args.tokens_per_day, args.seed)
    print('SYNTHETIC ONLY:')
    for level in [2, 3, 5, 10, 20]:
        print(f'share max_x>={level}: {(labels.max_x >= level).mean():.4%}')
    print(f'median final_x: {labels.final_x.median():.6f}')


if __name__ == '__main__':
    main()
