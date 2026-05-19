# Reviewer-Required Diagnostics

This report records diagnostics added in response to the reviewer-style critique.

## Target Bound Audit

| target   |   valid_raw |   kept_0_1 |   outside_0_1 |   outside_pct |   min_raw |   max_raw |
|:---------|------------:|-----------:|--------------:|--------------:|----------:|----------:|
| PHIF     |       47314 |      47314 |             0 |        0.0000 |    0.0000 |    0.4033 |
| SW       |       47314 |      47314 |             0 |        0.0000 |    0.0130 |    1.0000 |
| VSH      |       45100 |      44729 |           371 |        0.0082 |   -0.2480 |    3.6543 |

## Mendeley Dataset Summary

|   WELLNUM | WELL_NAME                        | DOMAIN   |   samples |   depth_min |   depth_max |   phit_min |   phit_mean |   phit_max |   DTC_missing |   DTS_missing |   GR_missing |   NPHIL_missing |   RESD_LOG10_missing |   RHOB_missing |
|----------:|:---------------------------------|:---------|----------:|------------:|------------:|-----------:|------------:|-----------:|--------------:|--------------:|-------------:|----------------:|---------------------:|---------------:|
|         0 | Gulf of Mexico lower Purple sand | GOM      |        59 |  15715.0000 |  15744.0000 |     0.1674 |      0.2499 |     0.2894 |        0.0000 |        0.0000 |       0.0000 |          0.0000 |               0.0000 |         0.0000 |
|         1 | Dutch North Sea Westphalian C    | DNS      |       160 |   3875.0748 |   3984.9552 |     0.0000 |      0.0603 |     0.1680 |        0.0000 |        0.0000 |       0.0000 |          0.0000 |               0.0000 |         0.0000 |

## SPWLA Cross-Well Mean, SD, and Approximate 95% CI

| method_label          |   coverage_mean |   coverage_sd |   p10_well_coverage_mean |   worst_well_coverage_mean |   width_mean |   winkler_mean |   winkler_sd |   winkler_ci95_low |   winkler_ci95_high |   rank_winkler |
|:----------------------|----------------:|--------------:|-------------------------:|---------------------------:|-------------:|---------------:|-------------:|-------------------:|--------------------:|---------------:|
| WL-PCQR               |          0.9262 |        0.0229 |                   0.8627 |                     0.8273 |       0.1212 |         0.1589 |       0.0701 |             0.1131 |              0.2048 |              1 |
| Global RF-CQR         |          0.9364 |        0.0211 |                   0.8944 |                     0.8542 |       0.1535 |         0.1864 |       0.0889 |             0.1283 |              0.2445 |              2 |
| Source-only RF-CQR    |          0.9029 |        0.0049 |                   0.8018 |                     0.6396 |       0.1422 |         0.2010 |       0.0868 |             0.1443 |              0.2577 |              3 |
| NGBoost PI            |          0.9287 |        0.0122 |                   0.8906 |                     0.8628 |       0.1641 |         0.2204 |       0.1136 |             0.1461 |              0.2946 |              4 |
| ExtraTrees-PCQR       |          0.9407 |        0.0073 |                   0.8908 |                     0.8464 |       0.2109 |         0.2255 |       0.1074 |             0.1553 |              0.2956 |              5 |
| CatBoost Quantile+CQR |          0.9256 |        0.0121 |                   0.8844 |                     0.8559 |       0.2257 |         0.2689 |       0.1253 |             0.1871 |              0.3507 |              6 |
| HGB-CQR               |          0.9332 |        0.0075 |                   0.8812 |                     0.8542 |       0.2424 |         0.2759 |       0.1108 |             0.2036 |              0.3483 |              7 |
| LightGBM Quantile+CQR |          0.9429 |        0.0155 |                   0.9043 |                     0.8828 |       0.2544 |         0.2817 |       0.1364 |             0.1926 |              0.3708 |              8 |

## Reviewer-Requested Near-Neighbor Baselines

| method_label       |   coverage |   p10_well_coverage |   worst_well_coverage |   width |   winkler |   rmse |   rank_winkler |
|:-------------------|-----------:|--------------------:|----------------------:|--------:|----------:|-------:|---------------:|
| WL-PCQR            |     0.9262 |              0.8627 |                0.8273 |  0.1212 |    0.1589 | 0.0420 |              1 |
| kNN Local RF-CQR   |     0.9396 |              0.8865 |                0.8524 |  0.1522 |    0.1810 | 0.0510 |              2 |
| Global RF-CQR      |     0.9364 |              0.8944 |                0.8542 |  0.1535 |    0.1864 | 0.0510 |              3 |
| RF-Leaf QRF+CQR    |     0.9660 |              0.9378 |                0.9219 |  0.1637 |    0.1962 | 0.0510 |              4 |
| GR-Mondrian RF-CQR |     0.9605 |              0.9347 |                0.9132 |  0.1802 |    0.1995 | 0.0510 |              5 |

## Depth-Gap Stratified Calibration

|   gap_steps |   coverage |   p10_well_coverage |   worst_well_coverage |   width |   winkler |   rmse |
|------------:|-----------:|--------------------:|----------------------:|--------:|----------:|-------:|
|      0.0000 |     0.9262 |              0.8627 |                0.8273 |  0.1212 |    0.1589 | 0.0420 |
|      3.0000 |     0.9568 |              0.8667 |                0.7778 |  0.1335 |    0.1441 | 0.0313 |

## Proper-Split Selection Bias Diagnostic

| method_label         |   coverage |   p10_well_coverage |   worst_well_coverage |   width |   winkler |   rmse |
|:---------------------|-----------:|--------------------:|----------------------:|--------:|----------:|-------:|
| Proper-split WL-PCQR |     0.9796 |              0.9561 |                0.9366 |  0.2376 |    0.2528 | 0.0453 |

Interpretation: the new diagnostics are not new components of WL-PCQR. They are reviewer-risk controls for target-bound preprocessing, calibration representativeness, nearest-neighbor conformal alternatives, and selection-induced calibration bias.