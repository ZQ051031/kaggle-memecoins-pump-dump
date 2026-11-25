# Memecoins: Pump or Dump?

An offline, CPU-only baseline for **Memecoins: Pump or Dump? Predict Pump.fun
Launches on Solana**. It converts raw launch/trade data into point-in-time
features, models the probability of reaching different price multiples, and
chooses a take-profit level or skips each token. The standalone
[`solution.ipynb`](solution.ipynb) contains all runtime code and needs no checkout
or internet connection.

> **Important warnings**
>
> - This is a **community-run competition**. The advertised prizes are **not an
>   official Kaggle payout** and depend on the individual sponsor. Check the
>   current organiser rules, eligibility and payment conditions yourself.
> - **Synthetic numbers are NOT a competition score.** The generator plants
>   learnable signals; these results say nothing about a real trading edge.
> - Kaggle foundational rule **6b** says public sharing of competition code
>   should be on Kaggle forums/notebooks. **Check with the organisers/Kaggle
>   before relying on this public GitHub repository for eligibility.** An MIT
>   code licence does not override competition or data rules.
> - **Not financial advice.** This is a simulated competition payoff, not a
>   live trading system or a claim that these returns can be executed.

## Competition summary

The community competition is sponsored by **Dmitry Dryomov**, with an advertised
**$5,000** fund across **10 places**, plus discretionary community-contribution
awards:

| Place | Prize | Place | Prize |
|---|---:|---|---:|
| 1st | $1,250 | 6th | $400 |
| 2nd | $850 | 7th | $350 |
| 3rd | $600 | 8th | $250 |
| 4th | $500 | 9th | $200 |
| 5th | $450 | 10th | $150 |

Code/participation closes and freezes **30 November 2026, 23:59 UTC**. Final
scoring runs on **1–14 December 2026**; the final score is the **sum of 14 daily
PnLs** on days that occur after the freeze. The **Kaggle leaderboard is only a
sandbox**, not the final prize ranking. Prize eligibility also requires the bot
submission and account link below, positive official scores and public review.
Recheck the live competition pages for changes:
<https://www.kaggle.com/competitions/memecoins-pump-or-dump>.

## What is predicted, and what gets scored?

A pump.fun token first trades on a bonding curve, then migrates to PumpSwap.
The input snapshot is **migration +60 seconds**. The entry price is the last
AMM trade at or before that snapshot, using the pool's **effective reserves**,
not `sol / tok` for a particular trade.

- `mt`: migration Unix timestamp, UTC.
- `entry`: price at the snapshot; it is derived from the tape, not a label-file
  helper column.
- `max_x`: sustained maximum price multiple over `(snapshot, snapshot +1h]`,
  discarding the top `max(1%, 1)` prints, relative to entry.
- `final_x`: the last price in that outcome hour divided by entry.

For each token choose `tp=0` (skip), or `3 <= tp <= 100`. The exact payoff is:

```text
pnl(tp, max_x, final_x) =
    0                              if tp == 0
    tp - 1.3                       if max_x >= tp
    max(final_x - 1.3, -1)          otherwise
```

The objective is the sum of these payoffs, **not classification accuracy**.
With `max_x=10` and `final_x=0.5`:

| Chosen tp | 0 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|
| PnL | 0 | 3.7 | 8.7 | -0.8 |

The organiser's approximate reach base rates are 28% at 2x, 14% at 3x, 6% at
5x, 2% at 10x and 0.4% at 20x. Median final price is only a few hundredths of
entry; indiscriminate buying is ruinous. The synthetic target median is 0.06.

## Architecture

1. **Features:** projected parquet reads, one migration UTC day at a time.
   Curve execution/failure statistics, buyer concentration, launch/bundle
   metadata, first-minute AMM windows `[0,15)`, `[15,30)`, `[30,60]`, prices,
   effective quote depth, flow and sponsorship; then temporal creator, early
   buyer wallet and trailing-day regime history. There are 70 numeric features
   in the default fixture. `bundled_buys_share` uses a `1e15` raw-base-unit
   supply proxy. `curve_curve_age_s` is the actual prefixed curve-age column.
2. **Level classifiers:** three seeds averaged for each reach level
   `[3,5,7,10,20]`; LightGBM with small trees and regularisation. sklearn
   HistGradientBoosting is the forced/automatic fallback. Fewer than 15
   positives triggers lower-level probability scaling by training base rates
   (or the base rate itself at the first level).
3. **Calibration:** Platt logistic calibration of logit probabilities using
   temporal walk-forward OOF predictions. A cumulative minimum enforces
   non-increasing probabilities with increasing take level. Between levels,
   interpolation is linear in log probability versus log level; above 20x a
   training-estimated power-law tail is used.
4. **Expected value:** held payoff `max(final_x-1.3,-1)` is estimated among
   missed takes, in five training-OOF P3 quantile bins, with unconditional
   fallback. For each candidate take:
   `EV = P(reach)*(tp-1.3) + (1-P(reach))*E(held | missed)`.
   The fixed grid is `[3,4,5,6,7,8,10,12,15,20,30,50,100]`.
5. **Selection:** maximize EV, skip unless above a fixed margin, and apply
   selected exclusions. Available candidates are printed: high bundled share,
   high bundled amount, instant curve fill, deep AMM drawdown and high AMM top
   buyer concentration. Training-OOF quantiles define thresholds. Margin from
   `[0,.1,.25,.5,1,2]` and a greedy subset of exclusions are tuned on pooled
   training OOF PnL, then frozen before test decisions.

Walk-forward uses the last seven training UTC days, fitting strictly before
each day minus two hours. The table's `pnl` is **in-sample-tuned on training
OOF**. `default_pnl` is the EV policy with **margin 0 and no exclusions**, and
`baseline_pnl` takes 5x only when calibrated P5 exceeds its held-payoff
break-even probability. Default/baseline still use OOF-fitted calibration and
held-payoff estimates: neither is a nested, completely unbiased score.

No labels, fewer than 200 labels, no eligible temporal OOF folds or a modelling
exception produces all-zero takes. Missing tapes/metadata yield NaNs, and a
missing archive is optional. Missing `test.csv` uses sample-submission mints
with zero takes. An explicit data directory is authoritative. Output preserves
test mint order; the optional Kaggle output write tolerates permission errors.

## Data and repository layout

```text
solution.ipynb                  self-contained, generated, unexecuted notebook
run.py                          real/synthetic input pipeline CLI
src/features.py                 discovery, daily aggregation, temporal histories
src/model.py                    classifiers, calibration, walk-forward pipeline
src/scoring.py                  payoff, EV/held bins, frozen policy, validation
src/synthetic.py                documented planted signals and fixture generator
scripts/build_notebook.py       inline modules into notebook cells
scripts/check_notebook.py       banned import/call/mint-literal guardrail
scripts/make_synthetic_data.py  generator CLI
scripts/run_local.py            synthetic pipeline + hidden-answer evaluation
tests/                          contract, leaks, models, policy, lint, execution
data/synthetic/                 ignored input CSVs and daily flat parquet tapes
data/synthetic_answers/         ignored hidden test labels, NEVER notebook input
requirements.txt                local development/execution dependencies
LICENSE                         MIT code licence (not a data licence)
```

Inputs: `train.csv` and optional `train_archive.csv` have
`mint,mt,max_x,final_x`; `test.csv` has `mint,mt`; `sample_submission.csv` has
`mint,tp`. Launch/migration metadata come from `pumpfun_token_creation.csv` and
`pfamm_migrations.csv`; missing migration mint can be joined via bonding curve.
Flat tapes are `day_YYYY-MM-DD_{pumpfun,pumpswap}_all_swaps.parquet`; recursive
`trades/day=YYYY-MM-DD/` layout is also supported. Sample copies prefixed
`0*_sample_` are skipped. Only present columns are read via schema projection.
Archive training includes at most the configured preceding days with tapes.

Auto-discovery tries `PUMP_DATA_DIR`, then the two canonical competition Kaggle
mounts, then one/two-level `/kaggle/input` directories, then `./data`,
`./data/synthetic`, `../data`, choosing the first containing `test.csv`.

### Locally

Python 3.10/3.11 is intended. Install dependencies, generate fixtures and run:

```bash
python -m pip install -r requirements.txt
python scripts/make_synthetic_data.py --out data/synthetic
python scripts/run_local.py
python scripts/run_local.py --no-lightgbm
python run.py --data-dir data/synthetic --out submission.csv
python run.py --data-dir /path/to/competition --wf-days 7 --archive-days 30
```

Generator defaults: `--days 40 --tokens-per-day 80 --seed 0`. Approximately ten
days are archive, 29 are training, then a two-hour exclusion gap and one test
day. Hidden answers go to the **sibling** `<input-name>_answers` directory.
The synthetic module docstring documents quality/creator/bundle/failure/flow
signals; these deliberately make the artificial task learnable.

`run_pipeline(data_dir=None, out_paths=None, wf_days=7, archive_days=30,
use_lightgbm=True)` returns `(submission_df, walk_forward_df)`. Defaults write
`./submission.csv` and attempt `/kaggle/working/submission.csv`. The local
runner, unlike the notebook/pipeline, reads hidden synthetic answers afterwards.

### On Kaggle

1. Join the competition and accept its rules yourself.
2. Create a Kaggle notebook and upload/import `solution.ipynb`.
3. **Add Input → competition data** for `memecoins-pump-or-dump`.
4. Use **CPU**, turn **Internet off**, and run all cells.
5. Inspect the walk-forward diagnostics and submission summary. The notebook
   writes `submission.csv` in the working directory and attempts the canonical
   `/kaggle/working/submission.csv` path.

Alternatively, after configuring your own Kaggle CLI credentials and accepting
the rules, place a `kernel-metadata.json` beside `solution.ipynb`:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/memecoins-offline-baseline",
  "title": "Memecoins offline baseline",
  "code_file": "solution.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "competition_sources": ["memecoins-pump-or-dump"],
  "dataset_sources": [],
  "kernel_sources": []
}
```

```bash
kaggle kernels push -p /path/to/folder-containing-notebook-and-metadata
```

This example is a user action, not an action performed by this repository.

### Telegram submission — user steps, before freeze

1. Join on Kaggle and accept all competition rules.
2. Send **`solution.ipynb`** as a notebook file to
   **`@pumpfun_competition_bot`** before **30 November 2026, 23:59 UTC**.
3. Send **`/link`** to that bot. Follow its one-time handshake: make **one Kaggle
   submission from your account with the bot-provided code as the description**.
   Wait for link confirmation. Both notebook delivery and linking must happen
   before freeze for prize eligibility; the Kaggle sandbox alone is not enough.
4. Review sandbox/static-scan feedback, fix legitimate problems and resubmit
   before freeze. Keep your own final notebook/version record.

The bot sandbox is **4 CPU / 31 GB RAM / 1 hour**, with a static scan and review.
This repository does not message the bot, link accounts, submit on Kaggle, or
promise acceptance. Follow current organiser instructions if they differ.

## Leakage and chain-order safeguards

- **One-hour resolution:** a label is usable only at `mt_j + 3600 <= query.mt`.
  The query cannot contribute its own outcome. Fold labels additionally obey
  `mt < fold_day_start - 7200`.
- **Cumulative history:** sorted event-time cumulative sums/as-of lookups and
  searchsorted bounds, not whole-file outcome means. Regime windows are exactly
  `[query.mt-3600-86400, query.mt-3600]`, including boundaries; rolling medians
  obey the same event timeline. Creator launch counts use strictly earlier
  label-free launches. Wallet reputation uses first 20 distinct executed buys.
- **Independent decisions:** no test ranking, test quantiles, test-wide caps or
  batch allocation. Calibration, held bins, thresholds, exclusions and margin
  are fixed from training before any test decision.
- **Chain order:** tapes are never resorted by timestamp; first/last and early
  wallets retain file order. Failed curve rows do not count as executed trades
  or prices. Prices come from provided reserve-derived `new_price`; AMM depth
  adds virtual quote reserves.
- **Offline:** no network access, external outcome lookup, serialized models,
  hardcoded token addresses or hidden answers in the submitted notebook.
  The linter is a static guardrail, **not a proof of security or rule compliance**.

## Testing and notebook maintenance

Source modules are authoritative. Rebuild after changing runtime code:

```bash
python scripts/build_notebook.py
python scripts/check_notebook.py
pytest -q
PUMP_DATA_DIR=/absolute/path/to/data/synthetic \
  jupyter nbconvert --to notebook --execute solution.ipynb \
  --output /tmp/executed.ipynb
```

The tests compare rebuilt code cells, reject a notebook importing `requests`,
execute the notebook in an isolated temporary working directory with
`PUMP_DATA_DIR`, and validate submission mint set/order, columns, domain,
uniqueness and completeness. They also test the payoff example, monotonicity,
all five candidate feature names, missing inputs, modelling fallback, temporal
boundaries and invariance to future deletion/test-row permutation. The linter
scans **all code cells and `src/*.py`** for prohibited imports/calls and
base58-like mint literals, allowing only the WSOL constant.

## Synthetic results — SYNTHETIC ONLY

**NOT a competition score or a live-trading result.** Final default
`python scripts/run_local.py` run: seed 0, 40 days × 80 tokens, LightGBM,
seven walk-forward days. Feature build: **2.514s**, 3,200 tokens, 70 columns.
Total runner runtime: **35.214s**. All five exclusion candidate columns were
available. Selected margin **0.1**; selected exclusion
`curve_curve_age_s < 262.6`.

| Day | Tokens | Buys | Hits | Tuned PnL | Baseline PnL | Default PnL | Cumulative tuned PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-22 | 80 | 10 | 5 | 9.500000 | 7.100000 | 7.500000 | 9.500000 |
| 2026-09-23 | 80 | 11 | 5 | 4.841654 | 7.159087 | 4.841654 | 14.341654 |
| 2026-09-24 | 80 | 7 | 2 | 0.498436 | 3.498436 | -2.501564 | 14.840089 |
| 2026-09-25 | 80 | 9 | 0 | -8.509822 | -1.809822 | -10.473774 | 6.330267 |
| 2026-09-26 | 80 | 9 | 6 | 11.200000 | -3.686843 | 11.200000 | 17.530267 |
| 2026-09-27 | 80 | 6 | 3 | 10.266310 | 11.886856 | 8.420539 | 27.796577 |
| 2026-09-28 | 80 | 11 | 2 | -1.454349 | -1.454349 | -3.454349 | 26.342228 |

All tuned rows are **in-sample-tuned on training OOF**. Exact printed totals:

```text
OOF totals: PnL=26.342228; n buys=63; hits=23; default PnL=15.532505; baseline tp=5 PnL=22.693364
SYNTHETIC ONLY test: PnL=7.712719; n buys=12; hits=3
```

The default generated labels achieved reach shares **28.1875%, 13.3750%,
5.4062%, 1.8438%, 0.3125%** at 2/3/5/10/20x and median `final_x=0.059920`.
Local output was validated; `/kaggle/working` was not writable on this host, so
that optional write was caught. Runtime varies by host/load.

## Out-of-sample evaluation — SYNTHETIC ONLY

Not a competition score. The policy fitted on the seed-0 tuning fixture was frozen (LightGBM, margin 0.1, one exclusion rule `curve_curve_age_s <= 262.6`, same TP grid and calibration) and scored once on new synthetic blocks with new seeds and later, disjoint dates. Protocol pre-registered in `eval/PROTOCOL.md` (committed before any results); reproduce with `python scripts/evaluate_oos.py --blocks B1 B2`. Per-day tables: `eval/results_oos.md`, `eval/results_oos.json`.

| Block | Seed | Dates (UTC) | Days x tokens/day | PnL | Buys | Hits | Skip-all | tp=5 buy-all |
|---|---:|---|---|---:|---:|---:|---:|---:|
| B1 | 101 | 2026-10-01 – 2026-10-14 | 14 x 80 = 1,120 | +86.89 | 122 | 44 | 0 | -750.59 |
| B2 | 202 | 2026-10-15 – 2026-10-28 | 14 x 80 = 1,120 | +30.67 | 121 | 38 | 0 | -756.05 |
| B3 | 303 | 2026-10-29 – 2026-11-11 | 14 x 80 = 1,120 | +87.91 | 113 | 50 | 0 | -813.62 |

Both blocks are positive, so the stop line was not hit. No retuning was done: B2 is about a third of B1, so the edge varies a lot between blocks and gives nothing stable to tune toward. B3 was later scored once with the same unchanged policy as a measurement only (`python scripts/evaluate_oos.py --blocks B3 --allow-b3`; per-day table in `eval/results_oos_b3.md`). No tuning followed. The synthetic generator plants the signal the features look for, so these numbers show the pipeline works without leakage. They do not predict real performance.

## Limitations

- **Untested on real competition data and on the Kaggle image.** Synthetic
  schema tests are not a real-data audit; million-row partition memory/runtime
  remains unverified. Runtime code uses the sklearn threadpool controller to
  bound CPU use; installed sklearn-version compatibility should be checked.
- Calibration comes from **shorter-history OOF models**, while final classifiers
  see more history. Distribution shifts can break probability calibration.
- Policy margins/exclusions are **in-sample-tuned** on pooled OOF outcomes;
  calibration/held-payoff estimates also reuse those OOF labels. No nested
  policy-evaluation split or independent competition score is claimed.
- The edge is thin; tail events, seed noise and a few large takes can dominate
  totals. Synthetic quality signals do not establish an economic edge.
- Payoff is a **simulation**, not live trading: execution, latency, liquidity,
  slippage, market impact, outages and operational risks are not modelled.
- The delivered source/notebook is a baseline, not an organiser approval or
  eligibility guarantee. Recheck all deadlines, sharing restrictions and
  sponsor conditions before entering. **Not financial advice.**
