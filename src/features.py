"""File-order tape features and point-in-time histories (all times UTC).

Bundled buy share uses a 1e15 raw-base-unit supply proxy. Missing inputs produce
NaNs, not fabricated zero activity. Histories use a fixed .14 smoothing prior,
never an outcome-derived global prior. Feature frames retain mint as their key.
"""
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CURVE_COLUMNS = 'mint ts ts_rel direction sol tok wallet signing_wallet new_price failed slot tx_idx ix_idx'.split()
AMM_COLUMNS = 'mint ts_rel direction sol tok signing_wallet fee_payer new_price slot tx_idx pool_base_token_reserves_after pool_quote_token_reserves_after virtual_quote_reserves'.split()


def discover_data_dir(data_dir=None):
    """Return first candidate containing test.csv, or raise FileNotFoundError."""
    candidates = [data_dir, os.environ.get('PUMP_DATA_DIR'),
                  '/kaggle/input/competitions/memecoins-pump-or-dump',
                  '/kaggle/input/memecoins-pump-or-dump']
    candidates += sorted(Path('/kaggle/input').glob('*')) + sorted(Path('/kaggle/input').glob('*/*'))
    candidates += ['data', 'data/synthetic', '../data']
    for candidate in candidates:
        if candidate is not None and (Path(candidate)/'test.csv').is_file():
            return Path(candidate)
    raise FileNotFoundError('No data directory containing test.csv')


def discover_tapes(data_dir):
    """Map UTC day -> {curve: [paths], amm: [paths]} for both layouts."""
    result = {}
    for path in sorted(Path(data_dir).rglob('*.parquet')):
        if re.match(r'0.*_sample_', path.name):
            continue
        match = re.search(r'(?:day_|day=)(\d{4}-\d{2}-\d{2})', str(path))
        if match:
            kind = 'amm' if ('pumpswap' in str(path).lower() or 'amm' in str(path).lower()) else 'curve'
            result.setdefault(match[1], {'curve': [], 'amm': []})[kind].append(path)
    return result


def _csv(root, name):
    path = Path(root)/name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _ratio(a, b):
    return a / b if pd.notna(b) and b != 0 else np.nan


def _series(df, key):
    return df[key] if key in df else pd.Series(np.nan, index=df.index)


def _aggregate(frame, kind):
    """Vectorized per-day reductions; first/last always retain input order."""
    original = set(frame)
    g = frame.copy()
    index = pd.Index(g.mint.unique(), name='mint')
    f = pd.DataFrame(index=index)
    for col in set(CURVE_COLUMNS + AMM_COLUMNS) - set(g):
        g[col] = np.nan
    if kind == 'curve':
        bad = g[g.failed.eq(1)]
        gb = bad.groupby('mint', sort=False)
        f['n_failed'] = gb.size().reindex(index, fill_value=0)
        f['failed_frac'] = g.failed.ne(0).groupby(g.mint, sort=False).mean()
        f['failed_buy_share'] = bad.direction.eq('buy').groupby(bad.mint, sort=False).mean()
        f['unique_failed_wallets'] = gb.wallet.nunique().reindex(index, fill_value=0)
        g = g[g.failed.eq(0)].copy()
    else:
        g = g[g.ts_rel.between(0, 60)].copy()
    g['w'] = g.wallet if kind == 'curve' else g.signing_wallet
    g['buy'] = g.direction.isin(['buy', 'B'])
    g['sell'] = g.direction.isin(['sell', 'S'])
    g['buy_sol'] = g.sol.where(g.buy, 0.)
    g['sell_sol'] = g.sol.where(g.sell, 0.)
    groups = g.groupby('mint', sort=False)
    total = groups.sol.sum()
    buyvol = groups.buy_sol.sum()
    f['n_trades'] = groups.size().reindex(index, fill_value=0)
    f['volume_sol'] = total
    f['buy_share'] = buyvol / total.replace(0, np.nan)
    f['n_wallets'] = groups.w.nunique()
    wb = g[g.buy].groupby(['mint', 'w'], sort=False).sol.sum()
    shares = wb / wb.groupby(level=0, sort=False).transform('sum').replace(0, np.nan)
    f['top1_share'] = shares.groupby(level=0, sort=False).max()
    f['hhi'] = shares.pow(2).groupby(level=0, sort=False).sum()
    pairs = g.groupby(['mint', 'w'], sort=False)[['buy', 'sell']].max()
    f['roundtrip_wallet_frac'] = (pairs.buy & pairs.sell).groupby(level=0, sort=False).sum() / f.n_wallets.replace(0, np.nan)
    prices = g[g.new_price.gt(0)].groupby('mint', sort=False).new_price.agg(['first', 'last', 'max', 'min'])
    f['price_last'] = prices['last']
    if kind == 'curve':
        f['n_buys'] = groups.buy.sum()
        f['n_sells'] = groups.sell.sum()
        f['trades_per_wallet'] = f.n_trades / f.n_wallets.replace(0, np.nan)
        ordered = shares.sort_values(ascending=False, kind='stable')
        top = ordered.groupby(level=0, sort=False).cumcount().lt(3)
        f['top3_share'] = ordered[top].groupby(level=0, sort=False).sum()
        f['largest_buy_sol'] = g[g.buy].groupby('mint', sort=False).sol.max()
        f['largest_buy_share'] = f.largest_buy_sol / buyvol.replace(0, np.nan)
        f['curve_age_s'] = -groups.ts_rel.first()
        for name, mask in [('last10min_share', g.ts_rel.gt(-600)),
                           ('last60s_pre_share', g.ts_rel.ge(-60) & g.ts_rel.lt(0))]:
            f[name] = g.sol.where(mask, 0.).groupby(g.mint, sort=False).sum() / total.replace(0, np.nan)
        f['sell_volume_share'] = groups.sell_sol.sum() / total.replace(0, np.nan)
        f['n_sellers_frac'] = g[g.sell].groupby('mint', sort=False).w.nunique() / f.n_wallets.replace(0, np.nan)
    else:
        for name, lo, hi in [('w1', 0, 15), ('w2', 15, 30), ('w3', 30, 60)]:
            mask = g.ts_rel.ge(lo) & (g.ts_rel.le(hi) if name == 'w3' else g.ts_rel.lt(hi))
            volume = g.sol.where(mask, 0.).groupby(g.mint, sort=False).sum()
            f['volume_'+name] = volume
            f['buy_share_'+name] = g.buy_sol.where(mask, 0.).groupby(g.mint, sort=False).sum() / volume.replace(0, np.nan)
        f['momentum'] = np.log1p(f.volume_w3 / (f.volume_w1 + 1e-12))
        f['price_first'] = prices['first']
        for name in ['last', 'max', 'min']:
            f['price_'+name+'_rel'] = prices[name] / prices['first']
        f['drawdown'] = prices['last'] / prices['max'] - 1
        g['depth'] = g.pool_quote_token_reserves_after + g.virtual_quote_reserves
        depth = g.groupby('mint', sort=False).depth.agg(['first', 'last'])
        f['depth_first'], f['depth_last'] = depth['first'], depth['last']
        f['depth_growth'] = depth['last'] / depth['first']
        clustered = g.groupby(['mint', 'slot'], sort=False).sol.transform('size').ge(2)
        f['slot_clustering_share'] = g.sol.where(clustered, 0.).groupby(g.mint, sort=False).sum() / total.replace(0, np.nan)
        f['sponsored_share'] = g.fee_payer.ne(g.signing_wallet).groupby(g.mint, sort=False).mean()
        f['net_buy_sol'] = buyvol - groups.sell_sol.sum()
    if 'direction' not in original:
        for col in f:
            if any(x in col for x in ['buy', 'sell', 'top', 'hhi', 'roundtrip']):
                f[col] = np.nan
    if 'sol' not in original:
        for col in f:
            if any(x in col for x in ['volume', 'buy_share', 'top', 'hhi', 'momentum', 'largest', 'clustering', 'last10', 'last60', 'net_buy']):
                f[col] = np.nan
    if 'fee_payer' not in original and 'sponsored_share' in f:
        f['sponsored_share'] = np.nan
    if kind == 'curve' and 'failed' not in original:
        f.loc[:, :] = np.nan
    return f.add_prefix(kind+'_')


def build_features(tokens, data_dir):
    """Return (one-row-per-token feature frame, mint/wallet early-buyers table).

    Only partitions needed by tokens are read, one day at a time. No label
    columns are copied. Parquet projection intersects each actual schema.
    """
    tokens = tokens[['mint', 'mt']].drop_duplicates('mint').reset_index(drop=True)
    result = tokens[['mint']].copy()
    result['hour'] = pd.to_datetime(tokens.mt, unit='s', utc=True).dt.hour
    records, early = {}, []
    tapes = discover_tapes(data_dir)
    days = pd.to_datetime(tokens.mt, unit='s', utc=True).dt.strftime('%Y-%m-%d')
    for day in days.unique():
        needed = set(tokens.loc[days.eq(day), 'mint'])
        for kind, columns in [('curve', CURVE_COLUMNS), ('amm', AMM_COLUMNS)]:
            parts = []
            for path in tapes.get(day, {}).get(kind, []):
                available = pq.read_schema(path).names
                cols = [c for c in columns if c in available]
                if 'mint' not in cols:
                    continue
                part = pd.read_parquet(path, columns=cols)
                parts.append(part[part.mint.isin(needed)])
            if not parts:
                continue
            frame = pd.concat(parts, ignore_index=True)
            if kind == 'curve' and 'wallet' not in frame and 'signing_wallet' in frame:
                frame['wallet'] = frame.signing_wallet
            records.setdefault(kind, []).append(_aggregate(frame, kind))
            if kind == 'curve' and {'failed', 'direction', 'wallet'} <= set(frame):
                buyers = frame.loc[frame.failed.eq(0) & frame.direction.eq('buy'), ['mint', 'wallet']].dropna().drop_duplicates()
                early.append(buyers[buyers.groupby('mint', sort=False).cumcount().lt(20)])
    template = pd.concat([_aggregate(pd.DataFrame(columns=CURVE_COLUMNS), 'curve'),
                          _aggregate(pd.DataFrame(columns=AMM_COLUMNS), 'amm')], axis=1)
    numeric = template.reindex(tokens.mint).astype(float)
    for kind, parts in records.items():
        values = pd.concat(parts)
        numeric.loc[:, values.columns] = values.reindex(tokens.mint).to_numpy()
    result = pd.concat([result, numeric.reset_index(drop=True)], axis=1)
    result['price_jump'] = result.amm_price_last / result.curve_price_last
    creation = _csv(data_dir, 'pumpfun_token_creation.csv')
    meta = tokens.merge(creation, on='mint', how='left') if 'mint' in creation else tokens.copy()
    result['launch_log_age'] = np.log1p((meta.mt - _series(meta, 'created_ts')).clip(lower=0))
    for c in ['bundle_size', 'bundled_buys', 'bundled_buys_count', 'direct_pf_invocation']:
        result[c] = _series(meta, c)
    result['log_dev_balance'] = np.log1p(_series(meta, 'dev_balance') / 1e9)
    result['bundled_buys_share'] = _series(meta, 'bundled_buys') / 1e15
    migrations = _csv(data_dir, 'pfamm_migrations.csv')
    if 'mint' not in migrations and 'bonding_curve' in migrations and {'mint', 'curve_address'} <= set(creation):
        migrations = migrations.merge(creation[['mint', 'curve_address']], left_on='bonding_curve', right_on='curve_address', how='left')
    mig = tokens.merge(migrations, on='mint', how='left') if 'mint' in migrations else tokens.copy()
    result['migration_log_sol'] = np.log1p(_series(mig, 'sol_amount') / 1e9)
    return result, pd.concat(early, ignore_index=True) if early else pd.DataFrame(columns=['mint', 'wallet'])


def _prefix(events, queries, value_columns, delay=3600, strict=False):
    """Sorted per-key prefix counts/sums; no label aggregation over the future."""
    output = np.zeros((len(queries), 1+len(value_columns)))
    e = events.dropna(subset=['key']).sort_values(['key', 'mt'], kind='stable').copy()
    if e.empty or queries.empty:
        return output
    e['count'] = e.groupby('key', sort=False).cumcount()+1
    for col in value_columns:
        e[col] = e.groupby('key', sort=False)[col].cumsum()
    e['event_time'] = e.mt + delay
    left = queries[['key', 'mt']].copy()
    left['position'] = np.arange(len(left))
    left = left.dropna(subset=['key']).sort_values('mt', kind='stable')
    if left.empty:
        return output
    merged = pd.merge_asof(left, e[['key', 'event_time', 'count']+value_columns].sort_values('event_time', kind='stable'),
                           left_on='mt', right_on='event_time', by='key', direction='backward',
                           allow_exact_matches=not strict)
    output[merged.position.to_numpy()] = merged[['count']+value_columns].fillna(0).to_numpy()
    return output


def history_features(query_df, labeled_df, universe_df=None, early_buyers=None,
                     creation_df=None, prior=.14, smoothing=5.):
    """Point-in-time creator, wallet and trailing-day regime features.

    universe_df includes all known mint/mt pairs; creator may be supplied there
    or via creation_df. Labels resolve at mt+3600 (inclusive). Prior universe
    launches use strictly mt<query.mt. Inputs are never modified.
    """
    q = query_df[['mint', 'mt']].reset_index(drop=True)
    universe = (universe_df if universe_df is not None else query_df).copy()
    if creation_df is not None and {'mint', 'creator'} <= set(creation_df):
        creators = creation_df[['mint', 'creator']].drop_duplicates('mint')
    elif 'creator' in universe:
        creators = universe[['mint', 'creator']].drop_duplicates('mint')
    else:
        creators = pd.DataFrame(columns=['mint', 'creator'])
    labels = labeled_df.reindex(columns=['mint', 'mt', 'max_x', 'final_x']).copy()
    if labels.empty:
        labels = pd.DataFrame(columns=['mint', 'mt', 'max_x', 'final_x'])
    labels = labels.drop_duplicates('mint')
    labels['hit'] = (labels.max_x >= 3).astype(float)
    labels['hit5'] = (labels.max_x >= 5).astype(float)
    labels['logmax'] = np.log(labels.max_x.clip(lower=1e-12).astype(float))
    labels['logfinal'] = np.log(labels.final_x.clip(lower=1e-12).astype(float))
    qc = q.merge(creators, on='mint', how='left').rename(columns={'creator': 'key'})
    lc = labels.merge(creators, on='mint', how='left').rename(columns={'creator': 'key'})
    c = _prefix(lc, qc, ['hit', 'logmax'])
    uc = universe[['mint', 'mt']].drop_duplicates('mint').merge(creators, on='mint', how='left').rename(columns={'creator': 'key'})
    launches = _prefix(uc, qc, [], delay=0, strict=True)[:, 0]
    out = q[['mint']].copy()
    out['creator_resolved_count'] = c[:, 0]
    out['creator_hit_rate'] = (c[:, 1]+prior*smoothing)/(c[:, 0]+smoothing)
    out['creator_mean_log_max'] = np.divide(c[:, 2], c[:, 0], out=np.full(len(q), np.nan), where=c[:, 0]>0)
    out['creator_prior_launches'] = launches
    missing = qc.key.isna()
    out.loc[missing, out.columns[1:]] = np.nan
    for col in ['wallet_hit_rate', 'wallet_prior_appearances', 'wallet_with_history_frac']:
        out[col] = np.nan
    if early_buyers is not None and not early_buyers.empty:
        early = early_buyers[['mint', 'wallet']].drop_duplicates()
        eq = q.reset_index(names='query_index').merge(early, on='mint').rename(columns={'wallet': 'key'})
        el = labels.merge(early, on='mint').rename(columns={'wallet': 'key'})
        wc = _prefix(el, eq, ['hit'])
        eq['rate'] = (wc[:, 1]+prior*smoothing)/(wc[:, 0]+smoothing)
        eq['count'] = wc[:, 0]
        eq['seen'] = wc[:, 0]>0
        summary = eq.groupby('query_index', sort=False).agg(wallet_hit_rate=('rate', 'mean'), wallet_prior_appearances=('count', 'sum'), wallet_with_history_frac=('seen', 'mean'))
        out.loc[summary.index, summary.columns] = summary.to_numpy()
    labels = labels.sort_values('mt', kind='stable')
    times = labels.mt.to_numpy(dtype=float)
    cumulative = {c: np.r_[0., labels[c].cumsum().to_numpy()] for c in ['hit', 'hit5']}
    lo = np.searchsorted(times, q.mt.to_numpy()-3600-86400, side='left')
    hi = np.searchsorted(times, q.mt.to_numpy()-3600, side='right')
    n = hi-lo
    out['regime_count'] = n
    for col, source in [('regime_hit3', 'hit'), ('regime_hit5', 'hit5')]:
        out[col] = np.divide(cumulative[source][hi]-cumulative[source][lo], n,
                             out=np.full(len(q), np.nan), where=n>0)
    # Rolling median on a merged event/query timeline handles ties inclusively.
    events = labels[['mt', 'logfinal']].rename(columns={'mt': 'time', 'logfinal': 'value'}).copy()
    events['position'] = -1
    events['order'] = 0
    queries = pd.DataFrame({'time': q.mt.to_numpy()-3600, 'value': np.nan,
                            'position': np.arange(len(q)), 'order': 1})
    timeline = pd.concat([events, queries], ignore_index=True).sort_values(['time', 'order'], kind='stable')
    if len(timeline):
        series = pd.Series(timeline.value.to_numpy(), index=pd.to_datetime(timeline.time, unit='s'))
        med = series.rolling('86400s', closed='both').median().to_numpy()
        selected = timeline.position.ge(0).to_numpy()
        values = np.full(len(q), np.nan)
        values[timeline.loc[selected, 'position'].to_numpy(dtype=int)] = med[selected]
        out['regime_median_log_final'] = values
    else:
        out['regime_median_log_final'] = np.nan
    return out
