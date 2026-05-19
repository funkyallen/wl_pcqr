# Runtime Benchmark

- target: `PHIF`
- seed: `42`
- max_per_well: `160`
- local calibration: `stratified` at `0.2`
- excluded features: `DEPTH`

| method              | method_label          | family    |   wall_time_sec |   time_per_split_sec |   estimated_3target_3seed_min |   winkler |   coverage |   width |   worst_well_coverage |   rank_winkler |   rank_time |
|:--------------------|:----------------------|:----------|----------------:|---------------------:|------------------------------:|----------:|-----------:|--------:|----------------------:|---------------:|------------:|
| wl_pcqr             | WL-PCQR               | classical |          7.1906 |               0.7990 |                        1.0786 |    0.0652 |     0.9106 |  0.0432 |                0.8203 |              1 |           3 |
| global_rf_cqr       | Global RF-CQR         | classical |          6.4468 |               0.7163 |                        0.9670 |    0.0657 |     0.9262 |  0.0490 |                0.8594 |              2 |           1 |
| ngboost_cqr_signed  | NGBoost PI            | classical |         28.3355 |               3.1484 |                        4.2503 |    0.0745 |     0.9167 |  0.0469 |                0.8438 |              3 |           7 |
| extratrees_pcqr     | ExtraTrees-PCQR       | classical |          6.6334 |               0.7370 |                        0.9950 |    0.0911 |     0.9323 |  0.0815 |                0.8828 |              4 |           2 |
| lightgbm_cqr_signed | LightGBM Quantile+CQR | classical |         11.9021 |               1.3225 |                        1.7853 |    0.1043 |     0.9314 |  0.0914 |                0.8828 |              5 |           4 |
| catboost_cqr_signed | CatBoost Quantile+CQR | classical |         17.6170 |               1.9574 |                        2.6425 |    0.1051 |     0.9323 |  0.0872 |                0.8828 |              6 |           6 |
| hgb_cqr_signed      | HGB-CQR               | classical |         17.0270 |               1.8919 |                        2.5540 |    0.1313 |     0.9418 |  0.1185 |                0.8594 |              7 |           5 |
