import numpy as np
import pandas as pd
import pytest

from src.model import LevelModels, PlattCalibrator, run_pipeline
from src.features import build_features
from src.scoring import (DecisionPolicy, LEVELS, TP_GRID, monotone_probs, pnl,
                         reach_probability, validate_submission)
from src.synthetic import generate_synthetic


def test_payoff():
    np.testing.assert_allclose(pnl([0, 5, 10, 20], 10, .5), [0, 3.7, 8.7, -.8])
    assert pnl(5, 2, .01) == -1
    assert pnl(5, 5, .01) == 3.7
    assert pnl(0, 100, 100) == 0


def test_monotonic_and_tail():
    raw = np.array([[.2, .4, .1, .12, .01], [.8, .6, .4, .3, .2]])
    p = monotone_probs(raw)
    assert (np.diff(p, axis=1) <= 0).all()
    interpolated = np.stack([reach_probability(p, tp) for tp in TP_GRID], axis=1)
    assert (np.diff(interpolated, axis=1) <= 0).all()
    np.testing.assert_allclose(reach_probability(p, 5), p[:, 1])
    np.testing.assert_allclose(reach_probability(p, 40, 2), p[:, -1]/4)
    labels = pd.DataFrame({'max_x': [1., 30.]})
    calibrated = PlattCalibrator().fit(raw, labels).predict(raw)
    assert (np.diff(calibrated, axis=1) <= 0).all()


def test_per_row_policy_invariance():
    rng = np.random.default_rng(11)
    features = pd.DataFrame({'mint': [f'm{i}' for i in range(50)], 'bundled_buys': rng.random(50),
                             'amm_drawdown': -rng.random(50)})
    probabilities = monotone_probs(rng.uniform(.02, .8, (50, 5)))
    labels = pd.DataFrame({'max_x': rng.lognormal(1, 1, 50), 'final_x': rng.uniform(.01, .5, 50)})
    policy = DecisionPolicy().fit(features, probabilities, labels)
    expected = policy.predict(features, probabilities)
    np.testing.assert_array_equal(policy.predict(features.iloc[:20], probabilities[:20]), expected[:20])
    shuffled = rng.permutation(50)
    np.testing.assert_array_equal(policy.predict(features.iloc[shuffled], probabilities[shuffled]), expected[shuffled])
    for i in [0, 15, 49]:
        assert policy.decide_row(features.iloc[i].to_dict(), probabilities[i]) == expected[i]


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp('models')/'synthetic'
    generate_synthetic(root, days=7, tokens_per_day=50)
    return root


def test_pipeline_real_hgb(synthetic):
    sub, table = run_pipeline(synthetic, out_paths=[], wf_days=1, use_lightgbm=False)
    assert validate_submission(sub, pd.read_csv(synthetic/'test.csv'))
    assert not table.empty, 'unexpected modelling fallback'
    assert table.n_tokens.sum() == 50
    assert {'default_pnl', 'baseline_pnl'} <= set(table)


def test_all_exclusion_candidates_exist(synthetic):
    labels = pd.read_csv(synthetic/'train.csv')
    features, _ = build_features(labels, synthetic)
    p = np.tile([.2, .1, .05, .02, .004], (len(labels), 1))
    policy = DecisionPolicy().fit(features, p, labels)
    assert set(policy.available_candidates) == {
        'bundled_buys_share', 'bundled_buys', 'curve_curve_age_s', 'amm_drawdown', 'amm_top1_share'}


def test_forced_backend_and_sparse_levels():
    x = pd.DataFrame({'a': np.linspace(0, 1, 220), 'b': np.nan})
    labels = pd.DataFrame({'max_x': np.where(x.a > .8, 4., 1.)})
    model = LevelModels(use_lightgbm=False).fit(x, labels)
    assert model.backend == 'sklearn HGB'
    assert len(model.models[0]) == 3
    p = model.predict(x)
    assert (np.diff(p, axis=1) <= 0).all()
    assert np.isfinite(p).all()


def test_empty_and_missing_test_fallbacks(tmp_path):
    test = pd.DataFrame({'mint': ['b', 'a'], 'mt': [200, 100]})
    test.to_csv(tmp_path/'test.csv', index=False)
    sub, table = run_pipeline(tmp_path, out_paths=[])
    assert validate_submission(sub, test) and sub.tp.eq(0).all() and table.empty
    (tmp_path/'test.csv').unlink()
    test[['mint']].assign(tp=9).to_csv(tmp_path/'sample_submission.csv', index=False)
    sub, _ = run_pipeline(tmp_path, out_paths=[])
    assert sub.mint.tolist() == ['b', 'a'] and sub.tp.eq(0).all()


def test_missing_archive_and_tapes(tmp_path):
    # >=200 labels exercise modelling rather than the small-data fallback.
    train = pd.DataFrame({'mint': [f'm{i}' for i in range(300)], 'mt': np.arange(300)*2000,
                          'max_x': 1.1, 'final_x': .06})
    test = pd.DataFrame({'mint': ['z'], 'mt': [700000]})
    train.to_csv(tmp_path/'train.csv', index=False)
    test.to_csv(tmp_path/'test.csv', index=False)
    sub, table = run_pipeline(tmp_path, out_paths=[], wf_days=1, use_lightgbm=False)
    assert validate_submission(sub, test)
    assert not table.empty


def test_modelling_exception_returns_zeros(synthetic, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('intentional failure')
    monkeypatch.setattr(LevelModels, 'fit', broken)
    sub, _ = run_pipeline(synthetic, out_paths=[], wf_days=1)
    assert validate_submission(sub, pd.read_csv(synthetic/'test.csv')) and sub.tp.eq(0).all()


@pytest.mark.parametrize('values', [[1.], [101.], [np.nan], [np.inf]])
def test_validation_errors(values):
    test = pd.DataFrame({'mint': ['x']})
    with pytest.raises(ValueError):
        validate_submission(test.assign(tp=values), test)


def test_validation_order_and_columns():
    test = pd.DataFrame({'mint': ['x', 'y']})
    with pytest.raises(ValueError):
        validate_submission(test.iloc[::-1].assign(tp=0.), test)
    with pytest.raises(ValueError):
        validate_submission(test.assign(tp=0., extra=1), test)
