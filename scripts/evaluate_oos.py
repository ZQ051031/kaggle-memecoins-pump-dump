"""Pre-registered SYNTHETIC ONLY frozen-policy rotations; never retunes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_features, history_features
from src.model import fit_pipeline
from src.scoring import pnl, validate_submission
from src.synthetic import generate_synthetic

BLOCKS = {
    'B1': {'seed': 101, 'start_date': '2026-10-01', 'end_date': '2026-10-14', 'days': 14, 'tokens_per_day': 80},
    'B2': {'seed': 202, 'start_date': '2026-10-15', 'end_date': '2026-10-28', 'days': 14, 'tokens_per_day': 80},
    'B3': {'seed': 303, 'start_date': '2026-10-29', 'end_date': '2026-11-11', 'days': 14, 'tokens_per_day': 80},
}


def fixture_hashes(data_dir):
    """Canonical directory-relative byte hashes, independent of output location."""
    data_dir = Path(data_dir)
    answer_dir = data_dir.parent/(data_dir.name+'_answers')
    return {f'{canonical}/{path.name}': hashlib.sha256(path.read_bytes()).hexdigest()
            for canonical, directory in [('synthetic', data_dir), ('synthetic_answers', answer_dir)]
            for path in sorted(directory.iterdir()) if path.is_file()}


def fit_frozen_artifacts(data_dir=ROOT/'data/synthetic'):
    """Refit only seed-0 tuning inputs, retaining all fitted state in memory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        generate_synthetic(data_dir, seed=0, days=40, tokens_per_day=80)
    expected = json.loads((ROOT/'eval/tuning_seed0_sha256.json').read_text())
    if fixture_hashes(data_dir) != expected:
        raise ValueError('Tuning fixture differs from pre-refactor seed-0 bytes; refusing evaluation')
    return fit_pipeline(data_dir, wf_days=7, archive_days=30, use_lightgbm=True)


def ensure_block(name, definition, data_root):
    """Generate once, or validate a complete previously generated block fixture."""
    root = Path(data_root)/name
    manifest = root/'block_manifest.json'
    required = ['test.csv', 'sample_submission.csv', 'pumpfun_token_creation.csv', 'pfamm_migrations.csv']
    for day in pd.date_range(definition['start_date'], periods=definition['days']):
        required.extend(f'day_{day:%Y-%m-%d}_{kind}_all_swaps.parquet' for kind in ['pumpfun', 'pumpswap'])
    if not root.exists():
        generate_synthetic(root, days=definition['days'], tokens_per_day=definition['tokens_per_day'],
                           seed=definition['seed'], start_date=definition['start_date'],
                           id_namespace=name, block=True)
        manifest.write_text(json.dumps(definition, indent=2)+'\n')
    if not manifest.exists() or json.loads(manifest.read_text()) != definition:
        raise ValueError(f'{name}: missing/mismatched generation manifest')
    if not all((root/file).exists() for file in required):
        raise ValueError(f'{name}: incomplete input fixture')
    if any((root/file).exists() for file in ['train.csv', 'train_archive.csv', 'test_labels.csv']):
        raise ValueError(f'{name}: block input must not contain labels')
    return root


def frozen_parameters(artifacts):
    return {'backend': artifacts.model.backend, 'margin': artifacts.policy.margin,
            'rules': [list(rule) for rule in artifacts.policy.rules], 'alpha': artifacts.policy.alpha,
            'feature_columns': artifacts.feature_columns,
            'held_bin_edges': artifacts.policy.held.edges.tolist(),
            'held_payoffs': artifacts.policy.held.values.tolist()}


def rotation_features(artifacts, rows, base, block_labels, universe, early, creation, day_start):
    """Only prior resolved rotations can add outcomes to the original context."""
    prior = block_labels[block_labels.mt < day_start-7200]
    visible = pd.concat([artifacts.training_labels, prior], ignore_index=True)
    history = history_features(rows, visible, universe, early, creation).set_index('mint')
    return base.reindex(rows.mint).reset_index().merge(history.reset_index(), on='mint', how='left')


def evaluate_block(artifacts, name, definition, data_root):
    """Score a whole block without fitting or modifying any artifact."""
    root = ensure_block(name, definition, data_root)
    tokens = pd.read_csv(root/'test.csv')
    if list(tokens) != ['mint', 'mt'] or tokens.mint.duplicated().any() or tokens.isna().any().any():
        raise ValueError(f'{name}: invalid token universe')
    expected_days = pd.date_range(definition['start_date'], periods=definition['days'], tz='UTC')
    token_days = pd.to_datetime(tokens.mt, unit='s', utc=True).dt.floor('D')
    counts = token_days.value_counts().reindex(expected_days, fill_value=0)
    if len(tokens) != definition['days']*definition['tokens_per_day'] or not counts.eq(definition['tokens_per_day']).all():
        raise ValueError(f'{name}: unexpected dates or token count')
    if set(tokens.mint) & set(artifacts.universe.mint):
        raise ValueError(f'{name}: token IDs overlap the tuning universe')
    labels = pd.read_csv(root.parent/(root.name+'_answers')/'test_labels.csv')
    if labels.mint.duplicated().any() or set(labels.mint) != set(tokens.mint):
        raise ValueError(f'{name}: answers do not match input mints')
    aligned = tokens.merge(labels, on=['mint', 'mt'], validate='one_to_one')
    if len(aligned) != len(tokens) or aligned[['max_x', 'final_x']].isna().any().any():
        raise ValueError(f'{name}: invalid answer timestamps/outcomes')
    base, block_early = build_features(tokens, root)
    base = base.set_index('mint')
    universe = pd.concat([artifacts.universe, tokens], ignore_index=True)
    early = pd.concat([artifacts.early_buyers, block_early], ignore_index=True)
    creation = pd.concat([artifacts.creation, pd.read_csv(root/'pumpfun_token_creation.csv')], ignore_index=True)
    before = frozen_parameters(artifacts)
    days = []
    for day in expected_days:
        rows = tokens[token_days.eq(day)]
        features = rotation_features(artifacts, rows, base, labels, universe, early, creation, int(day.timestamp()))
        takes = artifacts.predict_features(features, individually=True)
        submission = rows[['mint']].assign(tp=takes)
        validate_submission(submission, rows)
        # Current-day outcomes are joined only after all decisions for that day.
        scored = submission.merge(labels, on='mint', validate='one_to_one')
        buys = scored.tp.gt(0)
        days.append({'day': day.strftime('%Y-%m-%d'), 'n_tokens': len(rows), 'n_buys': int(buys.sum()),
                     'n_hits': int((buys & scored.max_x.ge(scored.tp)).sum()),
                     'pnl': float(pnl(scored.tp, scored.max_x, scored.final_x).sum()),
                     'skip_all_pnl': 0., 'tp5_buy_all_pnl': float(pnl(5., scored.max_x, scored.final_x).sum())})
    if frozen_parameters(artifacts) != before:
        raise RuntimeError('Frozen parameters mutated during evaluation')
    totals = {col: sum(day[col] for day in days) for col in days[0] if col != 'day'}
    return {'block': name, 'definition': definition, 'totals': totals, 'days': days}


def validate_block_request(blocks, allow_b3=False):
    if 'B3' in blocks:
        if not allow_b3:
            raise ValueError('B3 is reserved and requires --allow-b3')
        if blocks != ['B3']:
            raise ValueError('B3 is evaluated alone, as a frozen-policy measurement only: --blocks B3 --allow-b3')
        return
    if blocks not in [['B1'], ['B1', 'B2']]:
        raise ValueError('Evaluate B1 first; the only supported requests are B1 or B1 B2')


def evaluate_requested(artifacts, blocks, data_root):
    validate_block_request(blocks, allow_b3=blocks == ['B3'])
    evaluated = []
    reason = 'Requested blocks complete; STOP and report. No retune performed.'
    for name in blocks:
        result = evaluate_block(artifacts, name, BLOCKS[name], data_root)
        evaluated.append(result)
        print(f'SYNTHETIC ONLY {name}: {result["totals"]}')
        if result['totals']['pnl'] <= 0:
            reason = f'{name} PnL <= 0: STOP. No further blocks or tuning.'
            break
    return evaluated, reason


def write_results(results, frozen, reason, out_dir, suffix=''):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {'warning': 'SYNTHETIC ONLY — not a competition score', 'protocol': 'eval/PROTOCOL.md',
               'registered_blocks': BLOCKS, 'frozen': frozen, 'results': results, 'stop_reason': reason}
    (out_dir/f'results_oos{suffix}.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    text = ['# SYNTHETIC ONLY — frozen-policy OOS results', '',
            'Not a competition score or evidence of live-trading returns.', '',
            'Protocol: `eval/PROTOCOL.md`. No refitting/retuning on block outcomes.', '',
            '## Registered blocks', '', '| Block | Seed | UTC dates | Tokens |', '|---|---:|---|---:|']
    for name, d in BLOCKS.items():
        text.append(f'| {name} | {d["seed"]} | {d["start_date"]}–{d["end_date"]} | {d["days"]*d["tokens_per_day"]} |')
    text += ['', '## Frozen fit', '', '```json', json.dumps(frozen, indent=2), '```']
    columns = ['day', 'n_tokens', 'n_buys', 'n_hits', 'pnl', 'skip_all_pnl', 'tp5_buy_all_pnl']
    for result in results:
        text += ['', f'## {result["block"]}', '', '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        for day in result['days']:
            text.append('| '+' | '.join(f'{day[c]:.6f}' if isinstance(day[c], float) else str(day[c]) for c in columns)+' |')
        text += ['', 'Totals: `'+json.dumps(result['totals'], sort_keys=True)+'`']
    note = ('B3: frozen-policy measurement only; no retune or refit on any block outcome.' if suffix
            else 'B3 remains reserved; no authorized retune has been performed.')
    text += ['', reason, '', note]
    (out_dir/f'results_oos{suffix}.md').write_text('\n'.join(text)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', nargs='+', choices=list(BLOCKS), default=['B1', 'B2'])
    parser.add_argument('--allow-b3', action='store_true', help='Required for B3, which runs alone as a frozen-policy measurement (no retune)')
    args = parser.parse_args(argv)
    try:
        validate_block_request(args.blocks, args.allow_b3)
    except ValueError as error:
        parser.error(str(error))
    print('SYNTHETIC ONLY — pre-registered frozen-policy evaluation')
    artifacts = fit_frozen_artifacts()
    frozen = frozen_parameters(artifacts)
    print('Frozen parameters:', json.dumps(frozen, sort_keys=True))
    results, reason = evaluate_requested(artifacts, args.blocks, ROOT/'data/oos')
    write_results(results, frozen, reason, ROOT/'eval', '_b3' if args.blocks == ['B3'] else '')
    print(reason)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
