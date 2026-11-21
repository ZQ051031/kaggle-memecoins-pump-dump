from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (AMM_COLUMNS, CURVE_COLUMNS, build_features,
                          discover_data_dir, discover_tapes, history_features)
from src.synthetic import generate_synthetic


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('fixtures')/'synthetic'
    labels = generate_synthetic(root, days=4, tokens_per_day=8)
    return root, labels


def test_schema(dataset):
    root, labels = dataset
    assert list(pd.read_csv(root/'train.csv')) == ['mint', 'mt', 'max_x', 'final_x']
    test = pd.read_csv(root/'test.csv')
    assert list(test) == ['mint', 'mt']
    assert list(pd.read_csv(root/'sample_submission.csv')) == ['mint', 'tp']
    assert not (root/'test_labels.csv').exists()
    assert (root.parent/'synthetic_answers'/'test_labels.csv').exists()
    assert pd.read_csv(root/'train.csv').mt.max() <= test.mt.min()-7200
    for day in discover_tapes(root).values():
        curve = pd.read_parquet(day['curve'][0])
        amm = pd.read_parquet(day['amm'][0])
        assert set(CURVE_COLUMNS) <= set(curve)
        assert set(AMM_COLUMNS) <= set(amm)
        assert {'holder_rewards', 'instruction_type', 'virtual_sol_balance_after', 'block_time'} <= set(curve)
        assert {'pool_base_token_reserves_before', 'lp_fee', 'can_boost', 'base_token', 'quote_token'} <= set(amm)
        assert (curve.loc[curve.failed.eq(1), ['sol', 'tok', 'new_price']] == 0).all().all()
        assert amm.ts_rel.between(0, 60).all()
        np.testing.assert_allclose(amm.new_price, (amm.pool_quote_token_reserves_after+amm.virtual_quote_reserves)/amm.pool_base_token_reserves_after)
    assert labels.final_x.median() < .2


def test_shape_and_missing(dataset, tmp_path):
    root, labels = dataset
    f, early = build_features(labels, root)
    assert len(f) == len(labels) and f.mint.is_unique
    assert len(f.columns) == 60
    assert early.groupby('mint').size().max() <= 20
    assert not {'max_x', 'final_x'} & set(f)
    h = history_features(labels, labels, labels, early, pd.read_csv(root/'pumpfun_token_creation.csv'))
    assert len(h) == len(f)
    missing, wallets = build_features(labels, tmp_path)
    assert missing.curve_n_trades.isna().all() and wallets.empty
    empty = history_features(labels, pd.DataFrame())
    assert empty.regime_count.eq(0).all()
    assert discover_data_dir(root) == root
    no_tokens, no_wallets = build_features(labels.iloc[:0], root)
    assert no_tokens.empty and no_wallets.empty
    assert history_features(labels.iloc[:0], labels).empty


def test_future_deletion(dataset, tmp_path):
    root, labels = dataset
    x = labels.iloc[18]
    kept = labels[labels.mt <= x.mt]
    f, early = build_features(labels, root)
    # Physically delete future tokens from every input, including same-day tapes.
    for path in root.iterdir():
        if path.suffix == '.parquet':
            frame = pd.read_parquet(path)
            frame[frame.mint.isin(kept.mint)].to_parquet(tmp_path/path.name, index=False)
        elif path.suffix == '.csv':
            frame = pd.read_csv(path)
            frame[frame.mint.isin(kept.mint)].to_csv(tmp_path/path.name, index=False)
    short, early_short = build_features(kept, tmp_path)
    pd.testing.assert_frame_equal(f[f.mint.eq(x.mint)].reset_index(drop=True), short[short.mint.eq(x.mint)].reset_index(drop=True))
    creation = pd.read_csv(root/'pumpfun_token_creation.csv')
    h = history_features(labels, labels, labels, early, creation)
    hs = history_features(kept, kept, kept, early_short, creation[creation.mint.isin(kept.mint)])
    pd.testing.assert_frame_equal(h[h.mint.eq(x.mint)].reset_index(drop=True), hs[hs.mint.eq(x.mint)].reset_index(drop=True))


def test_resolution_delay_and_ties():
    universe = pd.DataFrame({'mint': ['old', 'recent', 'q', 'future'], 'mt': [0, 3601, 7200, 9000], 'creator': ['c']*4})
    labels = universe.assign(max_x=[4., 100., 99., 1000.], final_x=[.1, 100., 99., 1000.])
    early = pd.DataFrame({'mint': universe.mint, 'wallet': ['w']*4})
    q = universe.iloc[[2]]
    h = history_features(q, labels, universe, early)
    assert h.creator_resolved_count.iloc[0] == 1
    assert h.wallet_prior_appearances.iloc[0] == 1
    assert h.regime_count.iloc[0] == 1
    assert h.creator_prior_launches.iloc[0] == 2
    labels.loc[labels.mint.eq('recent'), 'mt'] = 3600
    h = history_features(q, labels, universe, early)
    assert h.creator_resolved_count.iloc[0] == 2
    assert h.regime_count.iloc[0] == 2


def test_folder_projection_and_file_order(tmp_path):
    folder = tmp_path/'trades'/'day=2026-01-01'
    folder.mkdir(parents=True)
    pd.DataFrame({'mint': ['x', 'x'], 'ts_rel': [20, 10], 'new_price': [4., 2.]}).to_parquet(folder/'amm.parquet')
    pd.DataFrame({'mint': ['x'], 'new_price': [999.]}).to_parquet(folder/'00_sample_amm.parquet')
    tokens = pd.DataFrame({'mint': ['x'], 'mt': [1767225600]})
    f, _ = build_features(tokens, tmp_path)
    assert f.amm_price_last.iloc[0] == 2.
    assert f.amm_buy_share.isna().all()
    assert f.curve_n_trades.isna().all()
    assert len(discover_tapes(tmp_path)['2026-01-01']['amm']) == 1


def test_invalid_generation(tmp_path):
    with pytest.raises(ValueError):
        generate_synthetic(tmp_path, days=1)


def test_deterministic(dataset, tmp_path):
    _, labels = dataset
    other = generate_synthetic(tmp_path/'other', days=4, tokens_per_day=8)
    pd.testing.assert_frame_equal(labels, other)
