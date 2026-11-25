"""Temporal level models and an offline, graceful-fallback pipeline."""
from pathlib import Path
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import contextlib


def _thread_limit(n=4):
    """Limit BLAS/OpenMP threads when sklearn exposes a controller; otherwise no-op."""
    try:
        from sklearn.utils.parallel import _get_threadpool_controller
        return _get_threadpool_controller().limit(limits=n)
    except Exception:
        return contextlib.nullcontext()

from src.features import build_features, discover_data_dir, discover_tapes, history_features
from src.scoring import (LEVELS, DecisionPolicy, monotone_probs, pnl,
                         reach_probability, validate_submission)


def _matrix(features):
    return features.drop(columns=['mint'], errors='ignore').replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


class LevelModels:
    """Three-seed bagging with sparse-positive scaled lower-level fallback."""

    def __init__(self, use_lightgbm=True):
        self.use_lightgbm = use_lightgbm

    def fit(self, features, labels):
        self.models = []
        x = _matrix(features)
        self.rates = np.array([(labels.max_x >= level).mean() for level in LEVELS])
        self.backend = 'sklearn HGB'
        classifier = None
        if self.use_lightgbm:
            try:
                from lightgbm import LGBMClassifier
                classifier = LGBMClassifier
                self.backend = 'LightGBM'
            except ImportError:
                pass
        with _thread_limit(4):
            for level in LEVELS:
                y = (labels.max_x.to_numpy() >= level).astype(int)
                fitted = []
                if y.sum() >= 15 and (1-y).sum() >= 15:
                    for seed in [0, 1, 2]:
                        if classifier is not None:
                            estimator = classifier(num_leaves=15, learning_rate=.05, n_estimators=300,
                                min_child_samples=40, subsample=.8, bagging_freq=1,
                                colsample_bytree=.8, reg_lambda=1., n_jobs=4, verbosity=-1, random_state=seed)
                        else:
                            estimator = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=.05,
                                max_iter=300, l2_regularization=1., random_state=seed, early_stopping=False)
                        estimator.fit(x, y)
                        fitted.append(estimator)
                self.models.append(fitted)
        r10, r20 = self.rates[-2:]
        self.alpha = float(np.clip(np.log(r10/r20)/np.log(2), .2, 5)) if r20 > 0 and r10 > r20 else 1.5
        return self

    def predict(self, features):
        x = _matrix(features)
        result = np.empty((len(x), len(LEVELS)))
        with _thread_limit(4):
            for j, fitted in enumerate(self.models):
                if fitted:
                    result[:, j] = np.mean([m.predict_proba(x)[:, 1] for m in fitted], axis=0)
                elif j == 0:
                    result[:, j] = self.rates[j]
                else:
                    result[:, j] = result[:, j-1] * self.rates[j]/max(self.rates[j-1], 1e-12)
        return monotone_probs(result)


class PlattCalibrator:
    """Per-level logistic calibration fit only on temporal held-out predictions."""

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p/(1-p)).reshape(-1, 1)

    def fit(self, probabilities, labels):
        self.models = []
        for j, level in enumerate(LEVELS):
            y = (labels.max_x.to_numpy() >= level).astype(int)
            model = None
            if len(np.unique(y)) == 2:
                model = LogisticRegression(C=1., solver='lbfgs').fit(self._logit(probabilities[:, j]), y)
            self.models.append(model)
        return self

    def predict(self, probabilities):
        result = np.asarray(probabilities).copy()
        for j, model in enumerate(self.models):
            if model is not None:
                result[:, j] = model.predict_proba(self._logit(result[:, j]))[:, 1]
        return monotone_probs(result)


def _read(root, name, columns):
    path = root/name
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def _write_submission(submission, out_paths):
    if isinstance(out_paths, (str, Path)):
        out_paths = [out_paths]
    for path in out_paths:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(path, index=False)
        except OSError as error:
            print(f'Could not write {path}: {error}')


@dataclass
class FittedArtifacts:
    """In-memory frozen fit and original history context; never serialized."""

    model: LevelModels
    calibration: PlattCalibrator
    policy: DecisionPolicy
    feature_columns: list
    training_labels: pd.DataFrame
    universe: pd.DataFrame
    creation: pd.DataFrame
    early_buyers: pd.DataFrame
    base_features: pd.DataFrame
    walk_forward: pd.DataFrame

    def predict_features(self, features, individually=False):
        ordered = features.reindex(columns=self.feature_columns)
        probabilities = self.calibration.predict(self.model.predict(ordered))
        if individually:
            return np.array([self.policy.decide_row(row, probability)
                             for row, probability in zip(ordered.to_dict('records'), probabilities)])
        return self.policy.predict(ordered, probabilities)


def load_training_labels(data_dir, archive_days=30):
    """Read the same eligible train/archive labels used by the original pipeline."""
    root = Path(data_dir)
    train = _read(root, 'train.csv', ['mint', 'mt', 'max_x', 'final_x'])
    archive = _read(root, 'train_archive.csv', ['mint', 'mt', 'max_x', 'final_x'])
    if archive_days > 0 and not archive.empty and not train.empty:
        dates = pd.to_datetime(archive.mt, unit='s', utc=True).dt.strftime('%Y-%m-%d')
        tape_days = {day for day, parts in discover_tapes(root).items() if parts['curve'] or parts['amm']}
        archive = archive[(archive.mt >= train.mt.min()-archive_days*86400) & dates.isin(tape_days)]
        train = pd.concat([archive, train], ignore_index=True)
    train = train.drop_duplicates('mint').dropna(subset=['mt', 'max_x', 'final_x']).sort_values('mt').reset_index(drop=True)
    return train


def fit_pipeline(data_dir, wf_days=7, archive_days=30, use_lightgbm=True,
                 training_labels=None, test_tokens=None):
    """Fit once, returning artifacts; errors propagate for evaluation callers."""
    root = Path(data_dir)
    train = load_training_labels(root, archive_days) if training_labels is None else training_labels
    test = _read(root, 'test.csv', ['mint', 'mt']) if test_tokens is None else test_tokens
    if len(train) < 200 or test.empty or test.mt.isna().any():
        raise ValueError('Insufficient labels or missing test times for a frozen fit')
    universe = pd.concat([train[['mint', 'mt']], test[['mint', 'mt']]], ignore_index=True).drop_duplicates('mint')
    creation = _read(root, 'pumpfun_token_creation.csv', ['mint', 'creator'])
    start = time.perf_counter()
    base, early = build_features(universe, root)
    base = base.set_index('mint')

    def features_for(rows, visible):
        history = history_features(rows, visible, universe, early, creation).set_index('mint')
        return base.reindex(rows.mint).reset_index().merge(history.reset_index(), on='mint', how='left')

    training_features = features_for(train, train)
    print(f'Features: {len(universe)} tokens, {len(training_features.columns)-1} columns, {time.perf_counter()-start:.3f}s')
    days = np.sort((train.mt.to_numpy()//86400).astype(int))
    fold_days = np.unique(days)[-max(0, int(wf_days)):] if wf_days > 0 else []
    predictions, indices, day_names, oof_parts = [], [], [], []
    for day in fold_days:
        fit_mask = train.mt < day*86400-7200
        val_mask = (train.mt >= day*86400) & (train.mt < (day+1)*86400)
        if fit_mask.sum() < 200 or not val_mask.any():
            continue
        fit_labels = train[fit_mask]
        model = LevelModels(use_lightgbm).fit(training_features.loc[fit_mask], fit_labels)
        valid = train[val_mask]
        valid_features = features_for(valid, fit_labels)
        predictions.append(model.predict(valid_features))
        oof_parts.append(valid_features)
        indices.extend(valid.index.tolist())
        day_names.extend([pd.to_datetime(day*86400, unit='s', utc=True).strftime('%Y-%m-%d')]*len(valid))
    if not predictions:
        raise ValueError('No eligible temporal OOF folds; cannot tune safely')
    oof_labels = train.loc[indices].reset_index(drop=True)
    oof_features = pd.concat(oof_parts, ignore_index=True)
    raw = np.concatenate(predictions)
    calibration = PlattCalibrator().fit(raw, oof_labels)
    calibrated = calibration.predict(raw)
    final_model = LevelModels(use_lightgbm).fit(training_features, train)
    policy = DecisionPolicy(final_model.alpha).fit(oof_features, calibrated, oof_labels)
    default_policy = DecisionPolicy(final_model.alpha)
    default_policy.held = policy.held
    default_takes = default_policy.predict(oof_features, calibrated)
    takes = policy.predict(oof_features, calibrated)
    gains = pnl(takes, oof_labels.max_x.to_numpy(), oof_labels.final_x.to_numpy())
    held5 = policy.held.predict(calibrated)[:, 2]
    p5 = reach_probability(calibrated, 5., final_model.alpha)
    baseline = np.where(p5 > -held5/(3.7-held5), 5., 0.)
    report = pd.DataFrame({'day': day_names, 'n_tokens': 1, 'n_buys': takes>0,
        'n_hits': (takes>0) & (oof_labels.max_x.to_numpy() >= takes), 'pnl': gains,
        'baseline_pnl': pnl(baseline, oof_labels.max_x.to_numpy(), oof_labels.final_x.to_numpy()),
        'default_pnl': pnl(default_takes, oof_labels.max_x.to_numpy(), oof_labels.final_x.to_numpy())})
    table = report.groupby('day', sort=True).sum().reset_index()
    table['cum_pnl'] = table.pnl.cumsum()
    table['tuning'] = 'in-sample-tuned on training OOF'
    return FittedArtifacts(final_model, calibration, policy, training_features.columns.tolist(),
                           train, universe, creation, early, base, table)


def run_pipeline(data_dir=None, out_paths=None, wf_days=7, archive_days=30, use_lightgbm=True,
                 return_artifacts=False):
    """Return (submission, walk-forward table), with zero-take safety fallback.

    Fold histories see only that fold's eligible labels. Calibration and policy
    tune on pooled temporal OOF, so reported policy PnL is in-sample-tuned.
    No synthetic answer file is ever read here.
    With return_artifacts=True, append the in-memory fit (or None on no fit).
    """
    if out_paths is None:
        out_paths = ['submission.csv', '/kaggle/working/submission.csv']
    table = pd.DataFrame(columns=['day', 'n_tokens', 'n_buys', 'n_hits', 'pnl', 'cum_pnl', 'default_pnl', 'baseline_pnl'])
    submission = pd.DataFrame({'mint': pd.Series(dtype=str), 'tp': pd.Series(dtype=float)})
    artifacts = None
    try:
        try:
            root = Path(data_dir) if data_dir is not None else discover_data_dir()
        except FileNotFoundError:
            root = Path(data_dir) if data_dir else Path('data')
        test = _read(root, 'test.csv', ['mint', 'mt'])
        if not (root/'test.csv').exists():
            test = _read(root, 'sample_submission.csv', ['mint']).reindex(columns=['mint', 'mt'])
        submission = test[['mint']].assign(tp=0.)
        train = load_training_labels(root, archive_days)
        if len(train) < 200 or test.empty or test.mt.isna().any():
            print(f'Zero-take fallback: {len(train)} labeled rows or missing test times')
        else:
            artifacts = fit_pipeline(root, wf_days, archive_days, use_lightgbm, train, test)
            table = artifacts.walk_forward
            history = history_features(test, train, artifacts.universe, artifacts.early_buyers,
                                       artifacts.creation).set_index('mint')
            test_features = artifacts.base_features.reindex(test.mint).reset_index().merge(
                history.reset_index(), on='mint', how='left')
            submission['tp'] = artifacts.predict_features(test_features)
            validate_submission(submission, test)
            print(f'Backend: {artifacts.model.backend}; margin={artifacts.policy.margin:g}; exclusions={artifacts.policy.rules}')
    except Exception as error:
        print(f'Zero-take fallback: {type(error).__name__}: {error}')
        submission['tp'] = 0.
    _write_submission(submission, out_paths)
    return (submission, table, artifacts) if return_artifacts else (submission, table)
