"""Competition payoff and fixed, row-independent expected-value policy."""
import numpy as np
import pandas as pd

LEVELS = np.array([3., 5., 7., 10., 20.])
TP_GRID = np.array([3., 4., 5., 6., 7., 8., 10., 12., 15., 20., 30., 50., 100.])


def pnl(tp, max_x, final_x):
    """Exact Evaluation payoff, scalar or broadcasting numpy arrays."""
    tp, max_x, final_x = np.broadcast_arrays(tp, max_x, final_x)
    result = np.where(tp == 0, 0., np.where(max_x >= tp, tp-1.3, np.maximum(final_x-1.3, -1.)))
    return float(result) if result.ndim == 0 else result


def monotone_probs(probabilities):
    return np.minimum.accumulate(np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1-1e-8), axis=-1)


def reach_probability(probabilities, tp, alpha=1.5):
    """Log-log interpolation between levels; power-law extrapolation above 20."""
    p = monotone_probs(probabilities)
    t = float(tp)
    if t > LEVELS[-1]:
        return np.clip(p[..., -1]*(20/t)**alpha, 1e-8, 1-1e-8)
    j = int(np.clip(np.searchsorted(LEVELS, t, side='right')-1, 0, len(LEVELS)-2))
    weight = np.clip(np.log(t/LEVELS[j])/np.log(LEVELS[j+1]/LEVELS[j]), 0, 1)
    return np.exp((1-weight)*np.log(p[..., j])+weight*np.log(p[..., j+1]))


class HeldPayoff:
    """Conditional failed-take payoff table with training-fixed P3 bins."""

    def fit(self, labels, probabilities):
        p3 = np.asarray(probabilities)[:, 0]
        self.edges = np.unique(np.quantile(p3, np.linspace(0, 1, 6)[1:-1]))
        bins = np.searchsorted(self.edges, p3, side='right')
        held = np.maximum(labels.final_x.to_numpy()-1.3, -1.)
        fallback = float(held.mean()) if len(held) else -1.
        self.values = np.full((len(self.edges)+1, len(TP_GRID)), fallback)
        for j, tp in enumerate(TP_GRID):
            missed = labels.max_x.to_numpy() < tp
            unconditional = held[missed].mean() if missed.any() else fallback
            for b in range(len(self.edges)+1):
                mask = missed & (bins == b)
                self.values[b, j] = held[mask].mean() if mask.any() else unconditional
        return self

    def predict(self, probabilities):
        bins = np.searchsorted(self.edges, np.asarray(probabilities)[..., 0], side='right')
        return self.values[bins]


def expected_values(probabilities, held, alpha=1.5):
    p = np.stack([reach_probability(probabilities, t, alpha) for t in TP_GRID], axis=-1)
    return p*(TP_GRID-1.3)+(1-p)*held


class DecisionPolicy:
    """OOF-tuned thresholds/margin; inference never uses batch statistics."""

    def __init__(self, alpha=1.5):
        self.alpha = alpha
        self.margin = 0.
        self.rules = []

    @staticmethod
    def _excluded(features, rules):
        mask = np.zeros(len(features), dtype=bool)
        for col, threshold, high in rules:
            if col in features:
                v = features[col].to_numpy()
                mask |= v > threshold if high else v < threshold
        return mask

    def fit(self, features, probabilities, labels):
        self.held = HeldPayoff().fit(labels, probabilities)
        ev = expected_values(probabilities, self.held.predict(probabilities), self.alpha)
        best = ev.argmax(axis=1)
        scores = ev[np.arange(len(ev)), best]
        chosen = TP_GRID[best]
        gains = pnl(chosen, labels.max_x.to_numpy(), labels.final_x.to_numpy())
        candidates = []
        candidate_columns = [('bundled_buys_share', True), ('bundled_buys', True),
                             ('curve_curve_age_s', False), ('amm_drawdown', False), ('amm_top1_share', True)]
        self.available_candidates = [col for col, _ in candidate_columns if col in features and features[col].notna().any()]
        print(f'Available exclusion candidates: {self.available_candidates}')
        for col, high in candidate_columns:
            if col in features and features[col].notna().any():
                for quantile in ([.9, .95] if high else [.05, .1]):
                    candidates.append((col, float(features[col].quantile(quantile)), high))
        best_pnl = -np.inf
        for margin in [0., .1, .25, .5, 1., 2.]:
            rules = []
            active = scores > margin
            score = gains[active].sum()
            while True:
                options = [(gains[active & ~self._excluded(features, [r])].sum(), r) for r in candidates if r not in rules]
                if not options:
                    break
                value, rule = max(options, key=lambda pair: pair[0])
                if value <= score+1e-12:
                    break
                rules.append(rule)
                active &= ~self._excluded(features, [rule])
                score = value
            if score > best_pnl:
                best_pnl, self.margin, self.rules = score, margin, rules
        return self

    def predict(self, features, probabilities):
        ev = expected_values(probabilities, self.held.predict(probabilities), self.alpha)
        j = ev.argmax(axis=1)
        active = (ev[np.arange(len(ev)), j] > self.margin) & ~self._excluded(features, self.rules)
        return np.where(active, TP_GRID[j], 0.).astype(float)

    def decide_row(self, features, probabilities):
        """Pure single-token decision given a frozen fitted policy."""
        return float(self.predict(pd.DataFrame([features]), np.asarray(probabilities).reshape(1, -1))[0])


def validate_submission(submission, test):
    """Raise ValueError for any contract violation; otherwise return True."""
    if list(submission.columns) != ['mint', 'tp']:
        raise ValueError('submission columns must be mint,tp')
    if submission.mint.duplicated().any() or submission.isna().any().any():
        raise ValueError('NaNs or duplicate mints')
    if submission.mint.tolist() != test.mint.tolist():
        raise ValueError('mint order/set differs from test')
    values = pd.to_numeric(submission.tp, errors='coerce').to_numpy()
    if not np.isfinite(values).all() or not ((values == 0) | ((values >= 3) & (values <= 100))).all():
        raise ValueError('tp outside {0} U [3,100]')
    return True
