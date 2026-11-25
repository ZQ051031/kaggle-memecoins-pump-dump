# Pre-registration: frozen-policy synthetic OOS evaluation

**SYNTHETIC ONLY. Not a competition score or evidence of live-trading returns.**
This protocol is written before generating or scoring any registered OOS block.
The implementation may be tested on unrelated miniature fixtures, but B1, B2 and
B3 must not be scored until the user explicitly authorizes evaluation.

## Frozen artifact

At the start of each evaluation invocation, deterministically run the existing
fitting pipeline once on the original `data/synthetic` fixture:
`generate_synthetic(seed=0, days=40, tokens_per_day=80)`, migration dates
2026-08-21 through 2026-09-29. Use the existing defaults: seven OOF days,
30 archive days, and LightGBM when available (otherwise the existing HGB fallback,
explicitly reported). Do not use the original hidden test answers for fitting.

Keep only in memory (no pickle or other model serialization): final `LevelModels`,
`PlattCalibrator`, `DecisionPolicy` including its margin, rules, held-payoff table,
alpha, and the feature column order. Retain the original eligible training labels
(train plus eligible archive), original universe/creation metadata and early-buyer
table as history context. Original test outcomes never join that context.

Freeze all model hyperparameters, seeds, take-profit grid, calibration procedure,
held-payoff logic, margin/exclusion tuning logic and generator planted process.
The refactor must preserve original seed-0 generator bytes and the original
runner results: OOF PnL 26.342228, 63 buys, 23 hits; original test PnL 7.712719,
12 buys, 3 hits. OOF remains in-sample-tuned; it is not OOS evidence.

## Registered blocks

| Block | Seed | Migration dates (UTC, inclusive) | Size | Role |
|---|---:|---|---:|---|
| B1 | 101 | 2026-10-01 through 2026-10-14 | 14 days x 80 = 1120 | First OOS test |
| B2 | 202 | 2026-10-15 through 2026-10-28 | 14 days x 80 = 1120 | Reproduction, only if B1 > 0 |
| B3 | 303 | 2026-10-29 through 2026-11-11 | 14 days x 80 = 1120 | Reserved; do not use now |

Block IDs namespace all synthetic entity identifiers (including seed) so no
mint, creator, wallet, curve, pool, signature or sponsor collides with seed 0 or
another block. This intentionally prevents cross-block entity reputation carryover;
global regime history can still use the permitted labels. WSOL is a shared quote
asset, not a synthetic identity. Dates and migration token sets are disjoint.
Only start date and identifier names change; no random draws or planted process
parameters change. Each block's input directory has test-style mint/mt lists,
creation/migration CSVs and tapes. All block outcomes live only in its sibling
answers directory, never in notebook inputs. Files are under ignored `data/oos/`.

## Daily rotations and strict label visibility

Evaluate each block independently from the original fitted artifact/history.
B2 does not inherit B1 labels; B3 does not inherit B1/B2 labels. For each block
UTC migration day D, use tape features from that block's own tapes. History
features may use as labeled set **only the original eligible training labels plus
labels of tokens in this same block with `mt < D_start - 7200`**. The feature
function's additional `mt_j + 3600 <= query.mt` rule remains in force. The
label-free universe can include the original universe and this block's tokens,
but creator launch counts still use strictly earlier `mt`.

Block labels feed history and post-decision scoring only. Do not refit any
classifier, calibrator, held-payoff bin/table, alpha, threshold, margin or rule.
Each token is decided alone with the frozen policy and frozen feature order;
no test ranking, quantiles, caps, shared allocation or decisions depending on
later rows. Join outcomes only after decisions are fixed for that day.

## Metrics and reporting

For each evaluated block report its definition, total PnL, number of buys,
number of hits (`tp > 0` and `max_x >= tp`), and a per-day table with those
metrics and token count. Use the exact Evaluation payoff:

```
tp == 0:          0
max_x >= tp:      tp - 1.3
otherwise:        max(final_x - 1.3, -1)
```

Report skip-all (PnL zero) and tp=5-buy-all reference PnLs per day and in total.
Print frozen backend, margin, exclusions and alpha; record the feature order,
held-payoff table/bin edges and block definitions for audit. Write
`eval/results_oos.json` and `eval/results_oos.md`, prominently SYNTHETIC ONLY.
Do not create results files before an authorized evaluation. Do not cherry-pick
seeds, days, subsets or policy variants after seeing results.

## Stop rule and reserved block

Evaluate B1 first. **If B1 PnL <= 0, stop immediately: do not generate/score B2,
do not tune, and do not open B3.** If B1 > 0, B2 may be evaluated once, solely
as a reproduction with the same frozen artifact and independent history context.
After the authorized B1/B2 evaluation, STOP and report; no automatic retuning.

B3 is reserved and may be used ONLY if B1 and B2 are both strictly positive
**and a subsequent retune has been explicitly authorized and completed**. It
requires an explicit B3 flag, evidence of the prerequisite results and retune,
and a new protocol identifying the retuned in-memory fit before any access.
The current frozen-policy evaluator must refuse B3 even with the flag until
that separately authorized retune/artifact is implemented. No retune is part of
this task. Do not run any registered-block evaluation during implementation.

## Amendment (after B1/B2 were reported): B3 frozen-policy measurement

B1 and B2 were both positive and no retune was performed. With user authorization, B3 (seed 303, 2026-10-29 to 2026-11-11) is scored once with the same unchanged frozen policy, as a measurement only: `python scripts/evaluate_oos.py --blocks B3 --allow-b3`. Results go to `eval/results_oos_b3.{md,json}`. No tuning or refit follows, whatever the result.
