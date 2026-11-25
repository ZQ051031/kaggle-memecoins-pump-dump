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

## B1

| day | n_tokens | n_buys | n_hits | pnl | skip_all_pnl | tp5_buy_all_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-01 | 80 | 2 | 0 | -2.000000 | 0.000000 | -59.802725 |
| 2026-10-02 | 80 | 5 | 2 | 6.400000 | 0.000000 | -54.966685 |
| 2026-10-03 | 80 | 5 | 2 | 2.400000 | 0.000000 | -43.766773 |
| 2026-10-04 | 80 | 13 | 6 | 9.200000 | 0.000000 | -50.962472 |
| 2026-10-05 | 80 | 10 | 4 | 4.800000 | 0.000000 | -50.923483 |
| 2026-10-06 | 80 | 9 | 4 | 1.800000 | 0.000000 | -58.133562 |
| 2026-10-07 | 80 | 10 | 3 | 7.740920 | 0.000000 | -55.647544 |
| 2026-10-08 | 80 | 7 | 4 | 10.800000 | 0.000000 | -60.678342 |
| 2026-10-09 | 80 | 14 | 4 | 7.843828 | 0.000000 | -51.295856 |
| 2026-10-10 | 80 | 7 | 1 | -2.300000 | 0.000000 | -51.250343 |
| 2026-10-11 | 80 | 8 | 3 | 11.105787 | 0.000000 | -59.367983 |
| 2026-10-12 | 80 | 9 | 4 | 20.407163 | 0.000000 | -39.946882 |
| 2026-10-13 | 80 | 8 | 4 | 12.800000 | 0.000000 | -54.262605 |
| 2026-10-14 | 80 | 15 | 3 | -4.108401 | 0.000000 | -59.584135 |

Totals: `{"n_buys": 122, "n_hits": 44, "n_tokens": 1120, "pnl": 86.88929647753442, "skip_all_pnl": 0.0, "tp5_buy_all_pnl": -750.5893895741324}`

## B2

| day | n_tokens | n_buys | n_hits | pnl | skip_all_pnl | tp5_buy_all_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-15 | 80 | 2 | 1 | 0.700000 | 0.000000 | -50.214347 |
| 2026-10-16 | 80 | 7 | 3 | 1.100000 | 0.000000 | -59.481391 |
| 2026-10-17 | 80 | 8 | 4 | 9.009497 | 0.000000 | -39.397253 |
| 2026-10-18 | 80 | 10 | 5 | 12.025413 | 0.000000 | -59.725828 |
| 2026-10-19 | 80 | 12 | 5 | 4.127117 | 0.000000 | -50.285534 |
| 2026-10-20 | 80 | 6 | 1 | -3.300000 | 0.000000 | -45.673795 |
| 2026-10-21 | 80 | 8 | 1 | -3.170810 | 0.000000 | -59.839235 |
| 2026-10-22 | 80 | 8 | 1 | 2.084656 | 0.000000 | -61.519067 |
| 2026-10-23 | 80 | 13 | 5 | 4.775570 | 0.000000 | -47.656374 |
| 2026-10-24 | 80 | 15 | 5 | -1.500000 | 0.000000 | -55.476510 |
| 2026-10-25 | 80 | 7 | 1 | -2.845979 | 0.000000 | -65.999179 |
| 2026-10-26 | 80 | 9 | 0 | -8.681833 | 0.000000 | -59.351099 |
| 2026-10-27 | 80 | 7 | 3 | 10.244766 | 0.000000 | -51.321935 |
| 2026-10-28 | 80 | 9 | 3 | 6.100000 | 0.000000 | -50.112780 |

Totals: `{"n_buys": 121, "n_hits": 38, "n_tokens": 1120, "pnl": 30.668397301160155, "skip_all_pnl": 0.0, "tp5_buy_all_pnl": -756.0543275882253}`

Requested blocks complete; STOP and report. No retune performed.

B3 remains reserved; no authorized retune has been performed.
