"""Deterministic SYNTHETIC ONLY fixture, not a proxy for competition performance.

Quality z is Gaussian plus a good-creator effect (15% repeat creators). Quality
increases unique curve buyers, largest buys, AMM buy share and late volume;
it reduces failures and bundles. Low-quality launches fill instantly, bundle
heavily and draw down. A noisy Gaussian quality score is transformed through
piecewise log-quantiles to a heavy tail with target exceedances 28/14/6/2/0.4%
at 2/3/5/10/20. Final multiples have median about .06 and correlate with quality.
Bundled amounts are raw token base units (supply proxy 1e15).
"""

from pathlib import Path
import re
from statistics import NormalDist

import numpy as np
import pandas as pd

WSOL = 'So11111111111111111111111111111111111111112'


def _namespace_frame(frame, prefix):
    if prefix:
        for col in ['mint', 'creator', 'wallet', 'signing_wallet', 'fee_payer',
                    'pool', 'base_token', 'quote_token', 'bonding_curve',
                    'curve_address', 'user', 'signature']:
            if col in frame:
                frame[col] = frame[col].where(frame[col].eq(WSOL), prefix + frame[col])
    return frame


def generate_synthetic(out, days=40, tokens_per_day=80, seed=0,
                       start_date='2026-08-21', id_namespace=None, block=False):
    """Write daily tapes and CSVs; return all labels (including hidden test).

    Answers are written to the sibling ``<out.name>_answers`` directory only.
    ``days`` includes one test day and must be at least three unless block=True.
    Blocks put every mint/mt in test.csv and every label only in answers.
    Nonzero seeds always namespace identities; an optional namespace additionally
    separates independent blocks with the same seed. Naming never changes RNG.
    """
    if days < (1 if block else 3) or tokens_per_day < 1:
        raise ValueError('days >= 3 (or >= 1 for blocks) and tokens_per_day >= 1 required')
    if id_namespace is not None and not re.fullmatch(r'[A-Za-z][A-Za-z0-9-]*', id_namespace):
        raise ValueError('id_namespace must be an alphanumeric/hyphen name starting with a letter')
    prefix = f'ns-{id_namespace or "default"}-seed-{seed}__' if id_namespace is not None or seed != 0 else ''
    out = Path(out)
    if block and any((out/name).exists() for name in ['train.csv', 'train_archive.csv']):
        raise ValueError('Block output must not contain existing training labels')
    out.mkdir(parents=True, exist_ok=True)
    answers = out.parent / (out.name + '_answers')
    answers.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels, creations, migrations = [], [], []
    stamp = pd.Timestamp(start_date)
    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    if stamp != stamp.normalize():
        raise ValueError('start_date must be midnight UTC')
    start = int(stamp.timestamp())
    normal = NormalDist()
    for day in range(days):
        curve, amm = [], []
        for k in range(tokens_per_day):
            i = day * tokens_per_day + k
            mint = f'token_{i:06d}'
            mt = start + day * 86400 + int((k + .5) * 78000 / tokens_per_day)
            creator = f'creator_{int(rng.integers(0, 20))}' if rng.random() < .15 else f'creator_unique_{i}'
            effect = .7 if creator in {'creator_0', 'creator_1', 'creator_2'} else 0
            z = rng.normal() + effect
            u = normal.cdf((.65 * z + np.sqrt(1 - .65**2) * rng.normal()) / 1.01)
            max_x = float(np.exp(np.interp(u, [0, .72, .86, .94, .98, .996, 1],
                                          np.log([1, 2, 3, 5, 10, 20, 150]))))
            final_x = float(min(max_x, .06 * np.exp(.35 * z + rng.normal(0, 1.15))))
            labels.append(dict(mint=mint, mt=mt, max_x=max_x, final_x=final_x))
            age = int(np.clip(np.exp(7 + .6 * z + rng.normal()), 90, 29 * 86400))
            bundle = float(1e15 * np.clip(.08 - .025 * z + rng.normal(0, .015), 0, .4))
            creations.append(dict(mint=mint, creator=creator, created_ts=mt-age,
                name=f'Synthetic {i}', symbol=f'S{i}', bundle_size=int(bundle/1e13),
                bundled_buys=bundle, bundled_buys_count=int(bundle/1e13), dev_balance=float(rng.lognormal(1, 1)*1e9),
                bundle_structure='[]', curve_address=f'curve_{i}', direct_pf_invocation=1, mayhem_mode=0))
            migrations.append(dict(mint=mint, pool=f'pool_{i}', bonding_curve=f'curve_{i}',
                sol_amount=float(rng.uniform(60, 100)*1e9), mint_amount=1e14, user=creator,
                signature=f'migration_{i}', slot=mt*2, tx_idx=0, block_time=pd.to_datetime(mt, unit='s', utc=True)))
            n = 90
            wallets = [f'wallet_{j}' for j in range(max(8, int(30 + 8*z)))]
            for j, rel in enumerate(np.linspace(-age, -1, n)):
                failed = int(rng.random() < np.clip(.08 - .025*z, .01, .3))
                buy = rng.random() < .72
                sol = 0. if failed else float(rng.lognormal(-1 + .15*z, .7))
                price = 0. if failed else .0002 * np.exp(j/n*.3)
                vb = 1e14 if not failed else 0.
                wallet = wallets[int(rng.integers(len(wallets)))]
                curve.append(dict(mint=mint, ts=mt+rel, ts_rel=rel, block_time=pd.to_datetime(mt+rel, unit='s', utc=True),
                    direction='buy' if buy else 'sell', sol=sol, tok=sol*1000, wallet=wallet, signing_wallet=wallet,
                    new_price=price, failed=failed, slot=int((mt+rel)*2), tx_idx=j, ix_idx=0,
                    virtual_sol_balance_after=price*vb, virtual_token_balance_after=vb,
                    instruction_type='buy' if buy else 'sell', min_amount_out=0, max_amount_in=int(sol*1e9),
                    fee=5000, provided_gas_fee=5000, consumed_gas=100000, tip_index=0, parent_program='synthetic_curve',
                    signature=f'curve_sig_{i}_{j}', quote_coin_amount=int(sol*1e9), base_coin_amount=int(sol*1e9),
                    holder_rewards=0, holder_rewards_bps=0))
            base_before, quote_before = 1e14, 2.7e10
            for j, rel in enumerate(np.linspace(0, 60, 36)):
                buy = rng.random() < np.clip(.5 + .13*z, .1, .9)
                sol = float(rng.lognormal(-1 + .35*z*rel/60, .5))
                price = .00027 * np.exp((.18*z + rng.normal(0, .06))*rel/60)
                base_after = base_before * (1 - .002 if buy else 1.002)
                virtual = 2e9
                quote_after = price*base_after - virtual
                wallet = wallets[int(rng.integers(len(wallets)))]
                amm.append(dict(mint=mint, ts=mt+rel, ts_rel=rel, block_time=pd.to_datetime(mt+rel, unit='s', utc=True),
                    direction='B' if buy else 'S', sol=sol, tok=sol*1000, signing_wallet=wallet,
                    fee_payer=wallet if rng.random() < .9 else 'sponsor', new_price=price, slot=int((mt+rel)*2),
                    tx_idx=j//2, pool=f'pool_{i}', base_token=mint, quote_token=WSOL,
                    pool_base_token_reserves_before=base_before, pool_base_token_reserves_after=base_after,
                    pool_quote_token_reserves_before=quote_before, pool_quote_token_reserves_after=quote_after,
                    virtual_quote_reserves=virtual, lp_fee=sol*.002, protocol_fee=sol*.0005,
                    coin_creator_fees=sol*.0005, lp_fee_bps=20, protocol_fee_bps=5, coin_creator_fee_bps=5,
                    signature=f'amm_sig_{i}_{j//2}', can_boost=False))
                base_before, quote_before = base_after, quote_after
        date = pd.to_datetime(start+day*86400, unit='s', utc=True).strftime('%Y-%m-%d')
        for name, rows in [('pumpfun', curve), ('pumpswap', amm)]:
            _namespace_frame(pd.DataFrame(rows), prefix).sort_values(['slot', 'tx_idx'], kind='stable').to_parquet(out/f'day_{date}_{name}_all_swaps.parquet', index=False)
    labels = _namespace_frame(pd.DataFrame(labels), prefix)
    test_start = start + (days-1)*86400
    archive_end = start + max(1, days//4)*86400
    if not block:
        labels[labels.mt < archive_end].to_csv(out/'train_archive.csv', index=False)
        labels[(labels.mt >= archive_end) & (labels.mt < test_start-7200)].to_csv(out/'train.csv', index=False)
    test = labels if block else labels[labels.mt >= test_start]
    test[['mint', 'mt']].to_csv(out/'test.csv', index=False)
    test.to_csv(answers/'test_labels.csv', index=False)
    test[['mint']].assign(tp=0.).to_csv(out/'sample_submission.csv', index=False)
    _namespace_frame(pd.DataFrame(creations), prefix).to_csv(out/'pumpfun_token_creation.csv', index=False)
    _namespace_frame(pd.DataFrame(migrations), prefix).to_csv(out/'pfamm_migrations.csv', index=False)
    return labels
