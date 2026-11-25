# SYNTHETIC ONLY — frozen-policy OOS results

Not a competition score or evidence of live-trading returns.

Protocol: `eval/PROTOCOL.md`. No refitting/retuning on block outcomes.

## Registered blocks

| Block | Seed | UTC dates | Tokens |
|---|---:|---|---:|
| B1 | 101 | 2026-10-01–2026-10-14 | 1120 |
| B2 | 202 | 2026-10-15–2026-10-28 | 1120 |
| B3 | 303 | 2026-10-29–2026-11-11 | 1120 |

## Frozen fit

```json
{
  "backend": "LightGBM",
  "margin": 0.1,
  "rules": [
    [
      "curve_curve_age_s",
      262.6,
      false
    ]
  ],
  "alpha": 2.5109619192773796,
  "feature_columns": [
    "mint",
    "hour",
    "curve_n_failed",
    "curve_failed_frac",
    "curve_failed_buy_share",
    "curve_unique_failed_wallets",
    "curve_n_trades",
    "curve_volume_sol",
    "curve_buy_share",
    "curve_n_wallets",
    "curve_top1_share",
    "curve_hhi",
    "curve_roundtrip_wallet_frac",
    "curve_price_last",
    "curve_n_buys",
    "curve_n_sells",
    "curve_trades_per_wallet",
    "curve_top3_share",
    "curve_largest_buy_sol",
    "curve_largest_buy_share",
    "curve_curve_age_s",
    "curve_last10min_share",
    "curve_last60s_pre_share",
    "curve_sell_volume_share",
    "curve_n_sellers_frac",
    "amm_n_trades",
    "amm_volume_sol",
    "amm_buy_share",
    "amm_n_wallets",
    "amm_top1_share",
    "amm_hhi",
    "amm_roundtrip_wallet_frac",
    "amm_price_last",
    "amm_volume_w1",
    "amm_buy_share_w1",
    "amm_volume_w2",
    "amm_buy_share_w2",
    "amm_volume_w3",
    "amm_buy_share_w3",
    "amm_momentum",
    "amm_price_first",
    "amm_price_last_rel",
    "amm_price_max_rel",
    "amm_price_min_rel",
    "amm_drawdown",
    "amm_depth_first",
    "amm_depth_last",
    "amm_depth_growth",
    "amm_slot_clustering_share",
    "amm_sponsored_share",
    "amm_net_buy_sol",
    "price_jump",
    "launch_log_age",
    "bundle_size",
    "bundled_buys",
    "bundled_buys_count",
    "direct_pf_invocation",
    "log_dev_balance",
    "bundled_buys_share",
    "migration_log_sol",
    "creator_resolved_count",
    "creator_hit_rate",
    "creator_mean_log_max",
    "creator_prior_launches",
    "wallet_hit_rate",
    "wallet_prior_appearances",
    "wallet_with_history_frac",
    "regime_count",
    "regime_hit3",
    "regime_hit5",
    "regime_median_log_final"
  ],
  "held_bin_edges": [
    0.025097899208850326,
    0.05415043233549469,
    0.11046510562680896,
    0.25036083774639745
  ],
  "held_payoffs": [
    [
      -0.9881020428903132,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037,
      -0.988420738170037
    ],
    [
      -0.9910229728868439,
      -0.9910229728868439,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359,
      -0.9911832769424359
    ],
    [
      -0.9695409329391204,
      -0.9651216426853714,
      -0.9651216426853714,
      -0.9651216426853714,
      -0.9649554546952481,
      -0.9649554546952481,
      -0.9649554546952481,
      -0.9649554546952481,
      -0.9649554546952481,
      -0.9649554546952481,
      -0.9649554546952481,
      -0.9649554546952481,
      -0.9649554546952481
    ],
    [
      -0.9845474103569797,
      -0.9843781978077414,
      -0.9828947557011009,
      -0.9838629770765103,
      -0.9840137903748606,
      -0.9841618108343526,
      -0.9795315265076477,
      -0.9795315265076477,
      -0.9795315265076477,
      -0.9758569084336791,
      -0.9758569084336791,
      -0.9758569084336791,
      -0.9758569084336791
    ],
    [
      -0.9676913324907813,
      -0.9715173589063468,
      -0.9697242496321367,
      -0.9716567443364685,
      -0.9311865352897101,
      -0.9325628045839158,
      -0.9332304995880355,
      -0.9345269947416659,
      -0.9357740996037294,
      -0.9369018978785342,
      -0.9369018978785342,
      -0.9374703492489977,
      -0.9374703492489977
    ]
  ]
}
```

## B3

| day | n_tokens | n_buys | n_hits | pnl | skip_all_pnl | tp5_buy_all_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-29 | 80 | 6 | 3 | 2.100000 | 0.000000 | -64.930760 |
| 2026-10-30 | 80 | 5 | 3 | 5.100000 | 0.000000 | -70.110905 |
| 2026-10-31 | 80 | 9 | 1 | -6.130369 | 0.000000 | -64.568657 |
| 2026-11-01 | 80 | 6 | 3 | 10.293389 | 0.000000 | -58.320677 |
| 2026-11-02 | 80 | 10 | 5 | 11.091539 | 0.000000 | -61.799761 |
| 2026-11-03 | 80 | 7 | 4 | 16.993328 | 0.000000 | -59.769309 |
| 2026-11-04 | 80 | 12 | 6 | 11.469622 | 0.000000 | -40.814279 |
| 2026-11-05 | 80 | 7 | 3 | 7.601586 | 0.000000 | -60.391108 |
| 2026-11-06 | 80 | 7 | 2 | -0.381542 | 0.000000 | -62.739025 |
| 2026-11-07 | 80 | 7 | 2 | 2.012219 | 0.000000 | -61.922287 |
| 2026-11-08 | 80 | 9 | 6 | 13.265063 | 0.000000 | -41.569351 |
| 2026-11-09 | 80 | 8 | 3 | 0.100000 | 0.000000 | -59.239545 |
| 2026-11-10 | 80 | 12 | 6 | 10.290899 | 0.000000 | -51.136433 |
| 2026-11-11 | 80 | 8 | 3 | 4.100000 | 0.000000 | -56.305925 |

Totals: `{"n_buys": 113, "n_hits": 50, "n_tokens": 1120, "pnl": 87.90573410584773, "skip_all_pnl": 0.0, "tp5_buy_all_pnl": -813.6180227973844}`

Requested blocks complete; STOP and report. No retune performed.

B3: frozen-policy measurement only; no retune or refit on any block outcome.
