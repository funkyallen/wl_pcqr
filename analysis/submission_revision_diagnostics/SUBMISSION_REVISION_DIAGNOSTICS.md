# Submission Revision Diagnostics

## VSH Outlier By Well

|   well |   raw_vsh_count |   removed_out_of_range_vsh |   removal_ratio |   min_raw_vsh |   max_raw_vsh |
|-------:|----------------:|---------------------------:|----------------:|--------------:|--------------:|
| 0.0000 |       1880.0000 |                    90.0000 |          0.0479 |       -0.2480 |        2.4601 |
| 1.0000 |       3534.0000 |                   281.0000 |          0.0795 |        0.0259 |        3.6543 |
| 2.0000 |       3166.0000 |                     0.0000 |          0.0000 |        0.0107 |        1.0000 |
| 3.0000 |       2533.0000 |                     0.0000 |          0.0000 |        0.0171 |        1.0000 |
| 4.0000 |       8679.0000 |                     0.0000 |          0.0000 |        0.0000 |        1.0000 |
| 5.0000 |       1765.0000 |                     0.0000 |          0.0000 |        0.0100 |        1.0000 |
| 6.0000 |      14077.0000 |                     0.0000 |          0.0000 |        0.0260 |        1.0000 |
| 7.0000 |       2164.0000 |                     0.0000 |          0.0000 |        0.0210 |        0.8760 |
| 8.0000 |       7302.0000 |                     0.0000 |          0.0000 |        0.0223 |        0.9136 |

## Main Protocol Split Sample Summary

| target   |   target_wells |   retained_per_well_min |   retained_per_well_max |   source_train_n_min |   source_train_n_max |   calibration_n_min |   calibration_n_max |   evaluation_n_min |   evaluation_n_max |
|:---------|---------------:|------------------------:|------------------------:|---------------------:|---------------------:|--------------------:|--------------------:|-------------------:|-------------------:|
| PHIF     |              9 |                     160 |                     160 |                 1280 |                 1280 |                  32 |                  32 |                128 |                128 |
| SW       |              9 |                     160 |                     160 |                 1280 |                 1280 |                  32 |                  32 |                128 |                128 |
| VSH      |              9 |                     160 |                     160 |                 1280 |                 1280 |                  32 |                  32 |                128 |                128 |

## Five Percent Calibration Sensitivity

|   frac | strategy   | method   | method_label   |   coverage |   p10_well_coverage |   worst_well_coverage |   width |   winkler |   rmse |
|-------:|:-----------|:---------|:---------------|-----------:|--------------------:|----------------------:|--------:|----------:|-------:|
| 0.0500 | contiguous | wl_pcqr  | WL-PCQR        |     0.7569 |              0.5202 |                0.4342 |  0.1273 |    0.3002 | 0.0542 |
| 0.0500 | random     | wl_pcqr  | WL-PCQR        |     0.8579 |              0.7351 |                0.6667 |  0.1482 |    0.2456 | 0.0505 |
| 0.0500 | stratified | wl_pcqr  | WL-PCQR        |     0.9080 |              0.8132 |                0.7500 |  0.1891 |    0.2385 | 0.0504 |

## Paired Winkler Bootstrap CI

| comparison                       |   paired_units |   mean_winkler_reduction |   ci95_low |   ci95_high |   wlpcqr_better_units |
|:---------------------------------|---------------:|-------------------------:|-----------:|------------:|----------------------:|
| Global RF-CQR minus WL-PCQR      |              9 |                   0.0275 |     0.0143 |      0.0398 |                     9 |
| kNN Local RF-CQR minus WL-PCQR   |              9 |                   0.0221 |     0.0101 |      0.0328 |                     6 |
| RF-Leaf QRF+CQR minus WL-PCQR    |              9 |                   0.0373 |     0.0216 |      0.0512 |                     9 |
| GR-Mondrian RF-CQR minus WL-PCQR |              9 |                   0.0406 |     0.0266 |      0.0531 |                     9 |

## Proximity Weight Summary

| target   |   splits |   fallback_rate |   fallback_count |   evaluation_n |   effective_n_min |   effective_n_p10 |   effective_n_median |   max_normalized_weight_p90 |
|:---------|---------:|----------------:|-----------------:|---------------:|------------------:|------------------:|---------------------:|----------------------------:|
| PHIF     |       27 |          0.0006 |                2 |           3456 |            1.0000 |            2.4626 |               5.3859 |                      0.6405 |
| SW       |       27 |          0.0041 |               14 |           3456 |            1.0000 |            3.9146 |              10.2095 |                      0.4550 |
| VSH      |       27 |          0.0009 |                3 |           3456 |            1.0000 |            2.7098 |               6.2238 |                      0.5487 |
| All      |       81 |          0.0018 |               19 |          10368 |            1.0000 |            3.0290 |               7.2731 |                      0.5481 |

## Hard-Well Petrophysical Summary

| target   |   well |   retained_n |   target_mean |   other_well_mean |   target_median |   other_well_median |   target_q10 |   target_q90 |   input_missing_rate |   other_input_missing_rate |   DEN_missing_rate |   DEN_other_missing_rate |   NEU_missing_rate |   NEU_other_missing_rate |   GR_missing_rate |   GR_other_missing_rate |   RDEP_LOG10_missing_rate |   RDEP_LOG10_other_missing_rate |   RMED_LOG10_missing_rate |   RMED_LOG10_other_missing_rate |
|:---------|-------:|-------------:|--------------:|------------------:|----------------:|--------------------:|-------------:|-------------:|---------------------:|---------------------------:|-------------------:|-------------------------:|-------------------:|-------------------------:|------------------:|------------------------:|--------------------------:|--------------------------------:|--------------------------:|--------------------------------:|
| PHIF     |      0 |         6885 |        0.1118 |            0.1417 |          0.0938 |              0.1420 |       0.0000 |       0.2703 |               0.4167 |                     0.1166 |             0.0000 |                   0.0000 |             0.0000 |                   0.0001 |            0.0000 |                  0.0000 |                    0.0000 |                          0.0058 |                    0.0000 |                          0.0000 |
| VSH      |      1 |         3253 |        0.4558 |            0.2876 |          0.3458 |              0.2329 |       0.2013 |       0.8096 |               0.1102 |                     0.1393 |             0.0452 |                   0.0206 |             0.0538 |                   0.0246 |            0.0000 |                  0.0000 |                    0.0037 |                          0.0077 |                    0.0037 |                          0.0021 |

## Hyperparameter Reproducibility Table

| component                     | setting                                   | value                      |
|:------------------------------|:------------------------------------------|:---------------------------|
| RandomForestRegressor         | n_estimators                              | 160                        |
| RandomForestRegressor         | min_samples_leaf                          | 8                          |
| RandomForestRegressor         | max_features                              | 0.75                       |
| ExtraTreesRegressor           | n_estimators                              | 180                        |
| ExtraTreesRegressor           | min_samples_leaf                          | 8                          |
| HistGradientBoostingRegressor | learning_rate                             | 0.05                       |
| HistGradientBoostingRegressor | max_iter                                  | 100-110                    |
| LightGBM                      | n_estimators / learning_rate              | 180 / 0.035                |
| CatBoost                      | iterations / depth / learning_rate        | 220 / 6 / 0.035            |
| NGBoost                       | distribution / estimators / learning_rate | Normal / 160 / 0.025       |
| All experiments               | seeds                                     | 42, 7, 123                 |
| Preprocessing                 | imputation                                | split-wise training median |
