# Baseline Comparison Suite

This suite uses the retained paper-facing baseline set.

Methods:

- WL-PCQR
- Global RF-CQR
- ExtraTrees-PCQR
- Source-only RF-CQR (cross-well only)
- NGBoost PI
- HGB-CQR
- LightGBM Quantile+CQR
- CatBoost Quantile+CQR

## Mendeley_GOM_DNS: cross_well

| method_label          |   coverage_mean |   p10_well_coverage_mean |   worst_well_coverage_mean |   width_mean |   winkler_mean |   rmse_mean |   rank_winkler |
|:----------------------|----------------:|-------------------------:|---------------------------:|-------------:|---------------:|------------:|---------------:|
| WL-PCQR               |          0.9448 |                   0.9353 |                     0.9297 |       0.0893 |         0.0980 |      0.0347 |              1 |
| NGBoost PI            |          0.9257 |                   0.8796 |                     0.8723 |       0.2941 |         0.3031 |      0.1151 |              2 |
| CatBoost Quantile+CQR |          0.9562 |                   0.8899 |                     0.8794 |       0.3509 |         0.3558 |      0.1546 |              3 |
| Global RF-CQR         |          0.9429 |                   0.8452 |                     0.8298 |       0.3576 |         0.3616 |      0.1404 |              4 |
| LightGBM Quantile+CQR |          0.9314 |                   0.8069 |                     0.7872 |       0.3598 |         0.3652 |      0.1244 |              5 |
| ExtraTrees-PCQR       |          0.9771 |                   0.9601 |                     0.9574 |       0.3919 |         0.3943 |      0.1521 |              6 |
| HGB-CQR               |          0.9143 |                   0.8413 |                     0.8298 |       0.3915 |         0.4049 |      0.1410 |              7 |
| Source-only RF-CQR    |          0.0030 |                   0.0011 |                     0.0000 |       0.0671 |         2.2473 |      0.1404 |              8 |

## Mendeley_GOM_DNS: within_well

| method_label          |   coverage_mean |   p10_well_coverage_mean |   worst_well_coverage_mean |   width_mean |   winkler_mean |   rmse_mean |   rank_winkler |
|:----------------------|----------------:|-------------------------:|---------------------------:|-------------:|---------------:|------------:|---------------:|
| Global RF-CQR         |          0.8250 |                   0.8151 |                     0.8132 |       0.0531 |         0.0813 |      0.0168 |              1 |
| WL-PCQR               |          0.8250 |                   0.8151 |                     0.8132 |       0.0531 |         0.0813 |      0.0168 |              1 |
| ExtraTrees-PCQR       |          0.8356 |                   0.7861 |                     0.7783 |       0.0631 |         0.0868 |      0.0185 |              3 |
| CatBoost Quantile+CQR |          0.8569 |                   0.7504 |                     0.7333 |       0.0526 |         0.0900 |      0.0158 |              4 |
| NGBoost PI            |          0.8341 |                   0.6802 |                     0.6556 |       0.0477 |         0.1066 |      0.0146 |              5 |
| HGB-CQR               |          0.8676 |                   0.8094 |                     0.8000 |       0.0866 |         0.1132 |      0.0267 |              6 |
| LightGBM Quantile+CQR |          0.8113 |                   0.7088 |                     0.6924 |       0.0638 |         0.1182 |      0.0141 |              7 |

## SPWLA_PDDA_2021: cross_well

| method_label          |   coverage_mean |   p10_well_coverage_mean |   worst_well_coverage_mean |   width_mean |   winkler_mean |   rmse_mean |   rank_winkler |
|:----------------------|----------------:|-------------------------:|---------------------------:|-------------:|---------------:|------------:|---------------:|
| WL-PCQR               |          0.9262 |                   0.8627 |                     0.8273 |       0.1212 |         0.1589 |      0.0420 |              1 |
| Global RF-CQR         |          0.9364 |                   0.8944 |                     0.8542 |       0.1535 |         0.1864 |      0.0510 |              2 |
| Source-only RF-CQR    |          0.9029 |                   0.8018 |                     0.6396 |       0.1422 |         0.2010 |      0.0523 |              3 |
| NGBoost PI            |          0.9287 |                   0.8906 |                     0.8628 |       0.1641 |         0.2204 |      0.0541 |              4 |
| ExtraTrees-PCQR       |          0.9407 |                   0.8908 |                     0.8464 |       0.2109 |         0.2255 |      0.0597 |              5 |
| CatBoost Quantile+CQR |          0.9256 |                   0.8844 |                     0.8559 |       0.2257 |         0.2689 |      0.0529 |              6 |
| HGB-CQR               |          0.9332 |                   0.8813 |                     0.8542 |       0.2424 |         0.2759 |      0.0568 |              7 |
| LightGBM Quantile+CQR |          0.9429 |                   0.9043 |                     0.8828 |       0.2544 |         0.2817 |      0.0517 |              8 |

## SPWLA_PDDA_2021: within_well

| method_label          |   coverage_mean |   p10_well_coverage_mean |   worst_well_coverage_mean |   width_mean |   winkler_mean |   rmse_mean |   rank_winkler |
|:----------------------|----------------:|-------------------------:|---------------------------:|-------------:|---------------:|------------:|---------------:|
| NGBoost PI            |          0.8513 |                   0.7994 |                     0.7611 |       0.1913 |         0.3850 |      0.0700 |              1 |
| Global RF-CQR         |          0.8849 |                   0.8293 |                     0.7799 |       0.2567 |         0.4125 |      0.0832 |              2 |
| WL-PCQR               |          0.8849 |                   0.8293 |                     0.7799 |       0.2567 |         0.4125 |      0.0832 |              2 |
| ExtraTrees-PCQR       |          0.8794 |                   0.8224 |                     0.7257 |       0.2713 |         0.4334 |      0.0937 |              4 |
| CatBoost Quantile+CQR |          0.8466 |                   0.7887 |                     0.7632 |       0.2458 |         0.4409 |      0.0837 |              5 |
| LightGBM Quantile+CQR |          0.8863 |                   0.8306 |                     0.8083 |       0.3459 |         0.4704 |      0.0735 |              6 |
| HGB-CQR               |          0.8771 |                   0.8121 |                     0.7521 |       0.3291 |         0.4858 |      0.0821 |              7 |
