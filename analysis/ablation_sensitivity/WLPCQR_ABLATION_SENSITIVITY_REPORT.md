# WL-PCQR Ablation and Sensitivity Report

## Mechanism Ablation

| method_label                           |   coverage |   p10_well_coverage |   worst_well_coverage |   width |   winkler |   rmse |
|:---------------------------------------|-----------:|--------------------:|----------------------:|--------:|----------:|-------:|
| Proximity location-scale recalibration |     0.9237 |              0.8516 |                0.7344 |  0.1207 |    0.1635 | 0.0387 |
| Proximity scale recalibration          |     0.9531 |              0.9062 |                0.8516 |  0.1542 |    0.1763 | 0.0510 |
| Raw proximity score                    |     0.8598 |              0.7656 |                0.7109 |  0.1300 |    0.1735 | 0.0510 |
| Uniform target-well CQR                |     0.9364 |              0.8828 |                0.7734 |  0.1535 |    0.1864 | 0.0510 |
| WL-PCQR selected path                  |     0.9262 |              0.8516 |                0.7344 |  0.1212 |    0.1589 | 0.0420 |

## WL-PCQR Selected Path Counts

| target   | selected_level   |   count |
|:---------|:-----------------|--------:|
| PHIF     | location-scale   |      15 |
| PHIF     | scale            |       3 |
| PHIF     | uniform          |       9 |
| SW       | location-scale   |      16 |
| SW       | scale            |       4 |
| SW       | uniform          |       7 |
| VSH      | location-scale   |      15 |
| VSH      | scale            |       6 |
| VSH      | uniform          |       6 |

## Sensitivity Summary

| experiment       | setting     |   coverage |   p10_well_coverage |   worst_well_coverage |   width |   winkler |   rmse |
|:-----------------|:------------|-----------:|--------------------:|----------------------:|--------:|----------:|-------:|
| depth_feature    | no_depth    |     0.9262 |              0.8516 |                0.7344 |  0.1212 |    0.1589 | 0.0420 |
| depth_feature    | with_depth  |     0.9325 |              0.8672 |                0.7891 |  0.1335 |    0.1675 | 0.0442 |
| nominal_coverage | alpha_0.05  |     0.9718 |              0.9219 |                0.8281 |  0.2035 |    0.2394 | 0.0449 |
| nominal_coverage | alpha_0.10  |     0.9262 |              0.8516 |                0.7344 |  0.1212 |    0.1589 | 0.0420 |
| nominal_coverage | alpha_0.20  |     0.8489 |              0.7578 |                0.6172 |  0.0820 |    0.1227 | 0.0401 |
| physical_bounds  | clip_to_0_1 |     0.9262 |              0.8516 |                0.7344 |  0.1212 |    0.1589 | 0.0420 |
| physical_bounds  | no_clip     |     0.9246 |              0.8516 |                0.7344 |  0.1268 |    0.1650 | 0.0420 |
