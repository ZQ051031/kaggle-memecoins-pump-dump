import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_oos as oos
from src.model import run_pipeline
from src.scoring import pnl
from src.synthetic import generate_synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def original_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('seed0-regression')/'synthetic'
    generate_synthetic(root)
    return root


def test_original_generator_bytes(original_fixture):
    # Captured from the original generator before any refactor/extensions.
    expected = json.loads((ROOT/'eval/tuning_seed0_sha256.json').read_text())
    assert oos.fixture_hashes(original_fixture) == expected


def test_eval_and_pipeline_artifacts_equal_on_original_seed0(original_fixture, tmp_path, monkeypatch):
    frozen = oos.fit_frozen_artifacts(original_fixture)
    submission, table, pipeline = run_pipeline(original_fixture, out_paths=[], return_artifacts=True)
    assert pipeline is not None
    assert oos.frozen_parameters(frozen) == oos.frozen_parameters(pipeline)
    assert frozen.policy.margin == pipeline.policy.margin
    assert frozen.policy.rules == pipeline.policy.rules
    np.testing.assert_array_equal(frozen.model.rates, pipeline.model.rates)
    pd.testing.assert_frame_equal(frozen.walk_forward, table)
    test = pd.read_csv(original_fixture/'test.csv')
    history = oos.history_features(test, frozen.training_labels, frozen.universe,
                                  frozen.early_buyers, frozen.creation).set_index('mint')
    features = frozen.base_features.reindex(test.mint).reset_index().merge(history.reset_index(), on='mint')
    np.testing.assert_array_equal(frozen.predict_features(features, individually=True), submission.tp)
    if frozen.model.backend == 'LightGBM':
        assert round(table.pnl.sum(), 6) == 26.342228
        assert table.n_buys.sum() == 63 and table.n_hits.sum() == 23
        labels = pd.read_csv(original_fixture.parent/'synthetic_answers/test_labels.csv')
        scored = submission.merge(labels, on='mint')
        assert round(pnl(scored.tp, scored.max_x, scored.final_x).sum(), 6) == 7.712719
        assert scored.tp.gt(0).sum() == 12
        assert (scored.tp.gt(0) & scored.max_x.ge(scored.tp)).sum() == 3
    # Exercise the evaluator only on an unrelated miniature fixture, never B1–B3.
    def no_refit(*args, **kwargs):
        raise AssertionError('No fit is allowed during block scoring')
    monkeypatch.setattr(type(frozen.model), 'fit', no_refit)
    monkeypatch.setattr(type(frozen.calibration), 'fit', no_refit)
    monkeypatch.setattr(type(frozen.policy), 'fit', no_refit)
    definition = {'seed': 909, 'start_date': '2027-05-01', 'end_date': '2027-05-02',
                  'days': 2, 'tokens_per_day': 3}
    result = oos.evaluate_block(frozen, 'FixtureCheck', definition, tmp_path)
    assert result['totals']['n_tokens'] == 6 and len(result['days']) == 2
    assert result['totals']['skip_all_pnl'] == 0
    assert oos.frozen_parameters(frozen) == oos.frozen_parameters(pipeline)


def _identities(root):
    values = set()
    columns = {'mint', 'creator', 'wallet', 'signing_wallet', 'fee_payer', 'pool',
               'base_token', 'bonding_curve', 'curve_address', 'user', 'signature'}
    for file in root.iterdir():
        frame = pd.read_parquet(file) if file.suffix == '.parquet' else pd.read_csv(file)
        for col in columns & set(frame):
            values.update(frame[col].dropna())
    return values


def test_miniature_block_namespaces_and_dates(tmp_path):
    # Unregistered fixture seeds/dates only; never generate B1, B2 or B3 here.
    original = tmp_path/'original'
    a, b = tmp_path/'fixtureA', tmp_path/'fixtureB'
    generate_synthetic(original, days=3, tokens_per_day=3)
    first = generate_synthetic(a, seed=901, days=2, tokens_per_day=3,
                               start_date='2027-01-01', id_namespace='FixtureA', block=True)
    second = generate_synthetic(b, seed=902, days=2, tokens_per_day=3,
                                start_date='2027-02-01', id_namespace='FixtureB', block=True)
    sets = [_identities(p) for p in [original, a, b]]
    for i in range(3):
        for j in range(i):
            assert sets[i].isdisjoint(sets[j])
    assert set(first.mint).isdisjoint(second.mint)
    assert set(first.mt//86400).isdisjoint(second.mt//86400)
    for root, labels in [(a, first), (b, second)]:
        assert list(pd.read_csv(root/'test.csv')) == ['mint', 'mt']
        assert len(pd.read_csv(root/'test.csv')) == 6
        assert not (root/'train.csv').exists() and not (root/'train_archive.csv').exists()
        assert not (root/'test_labels.csv').exists()
        pd.testing.assert_frame_equal(pd.read_csv(root.parent/(root.name+'_answers')/'test_labels.csv'), labels)


def test_registered_block_definitions_only():
    # Inspect dates/configuration, not outcomes or generated tapes.
    date_sets = []
    for name, definition in oos.BLOCKS.items():
        days = pd.date_range(definition['start_date'], periods=definition['days'])
        assert days[-1].strftime('%Y-%m-%d') == definition['end_date']
        assert len(days)*definition['tokens_per_day'] == 1120
        for earlier in date_sets:
            assert set(days).isdisjoint(earlier)
        assert days[0] > pd.Timestamp('2026-09-29')
        date_sets.append(set(days))
    assert len({d['seed'] for d in oos.BLOCKS.values()}) == 3


def test_namespace_and_date_do_not_change_draws(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    first = generate_synthetic(a, days=3, tokens_per_day=4)
    second = generate_synthetic(b, days=3, tokens_per_day=4,
                                start_date='2027-01-01', id_namespace='Other')
    np.testing.assert_array_equal(first[['max_x', 'final_x']], second[['max_x', 'final_x']])
    assert (second.mt-first.mt).nunique() == 1
    assert (second.mint == 'ns-Other-seed-0__'+first.mint).all()
    c = tmp_path/'nonzero'
    generate_synthetic(c, seed=901, days=1, tokens_per_day=2, block=True)
    assert pd.read_csv(c/'test.csv').mint.str.startswith('ns-default-seed-901__').all()


def test_rotation_history_cutoff(monkeypatch):
    captured = []
    def history(query, labeled, *args):
        captured.append(labeled)
        return query[['mint']].assign(history=0.)
    monkeypatch.setattr(oos, 'history_features', history)
    cutoff = 100000-7200
    block = pd.DataFrame({'mint': ['before', 'at', 'after'], 'mt': [cutoff-1, cutoff, cutoff+1],
                          'max_x': 3., 'final_x': .1})
    artifacts = SimpleNamespace(training_labels=pd.DataFrame({'mint': ['original'], 'mt': [0],
                                                             'max_x': [2.], 'final_x': [.1]}))
    query = pd.DataFrame({'mint': ['q'], 'mt': [100100]})
    base = query[['mint']].assign(f=1.).set_index('mint')
    oos.rotation_features(artifacts, query, base, block, query, pd.DataFrame(), pd.DataFrame(), 100000)
    assert captured[0].mint.tolist() == ['original', 'before']


@pytest.mark.parametrize('first_pnl,expected', [(0., ['B1']), (-1., ['B1']), (1., ['B1', 'B2'])])
def test_stop_rule_without_evaluating_registered_blocks(monkeypatch, tmp_path, first_pnl, expected):
    called = []
    def fake(artifacts, name, definition, root):
        called.append(name)
        return {'block': name, 'totals': {'pnl': first_pnl}, 'days': []}
    monkeypatch.setattr(oos, 'evaluate_block', fake)
    results, reason = oos.evaluate_requested(object(), ['B1', 'B2'], tmp_path)
    assert called == expected and [r['block'] for r in results] == expected
    assert 'STOP' in reason
    assert not list(tmp_path.iterdir())


def test_reserved_block_guards():
    for request, flag in [(['B3'], False), (['B1', 'B2', 'B3'], True), (['B2'], False)]:
        with pytest.raises(ValueError):
            oos.validate_block_request(request, flag)


def test_results_writer_only_with_mock_results(tmp_path):
    mock = {'block': 'FixtureOnly', 'definition': {'seed': 999},
            'days': [{'day': '2027-01-01', 'n_tokens': 2, 'n_buys': 1, 'n_hits': 0,
                      'pnl': -1., 'skip_all_pnl': 0., 'tp5_buy_all_pnl': -2.}],
            'totals': {'pnl': -1., 'n_tokens': 2, 'n_buys': 1, 'n_hits': 0,
                       'skip_all_pnl': 0., 'tp5_buy_all_pnl': -2.}}
    oos.write_results([mock], {'backend': 'mock', 'margin': .1, 'rules': []}, 'STOP', tmp_path)
    data = json.loads((tmp_path/'results_oos.json').read_text())
    assert data['results'][0]['block'] == 'FixtureOnly'
    assert 'SYNTHETIC ONLY' in (tmp_path/'results_oos.md').read_text()


def test_invalid_block_inputs(tmp_path):
    root = tmp_path/'existing'
    generate_synthetic(root, days=3, tokens_per_day=1)
    with pytest.raises(ValueError, match='existing training labels'):
        generate_synthetic(root, block=True, days=1)
    with pytest.raises(ValueError, match='id_namespace'):
        generate_synthetic(tmp_path/'invalid', id_namespace='bad_namespace')
