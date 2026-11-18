# Phase 3 Report: Rolling VaR backtest

Out-of-sample forecast dates: 2009-12-23 to 2025-12-31 (4030 days), identical for all set-ups.
Each forecast for day t uses only returns before t. Hits: realised loss > VaR.

Tests: Kupiec POF (chi2(1)), Christoffersen independence (chi2(1)) and conditional coverage
(chi2(2)). "REJECT" means the model is rejected at that significance level.
`es_ratio` = mean realised loss on violation days / mean forecast ES on those days
(descriptive; ~1 is good, >1 means ES understated). It is not a formal ES test.

### Primary: W=500, refit 21d
| model             | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |
|:------------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|
| Historical        | 95%     |          204 |      201.5 | 5.06%            |                19 |    0.857   | 0.01    | 0.036   | -               | REJECT         | -               | -              |       1.02 |
| Historical        | 99%     |           54 |       40.3 | 1.34%            |                 6 |    0.039   | 6.6e-05 | 4.2e-05 | REJECT          | REJECT         | -               | REJECT         |       1.07 |
| Normal            | 95%     |          204 |      201.5 | 5.06%            |                22 |    0.857   | 0.00073 | 0.003   | -               | REJECT         | -               | REJECT         |       1.24 |
| Normal            | 99%     |           88 |       40.3 | 2.18%            |                 9 |    6.6e-11 | 0.00012 | 3.3e-13 | REJECT          | REJECT         | REJECT          | REJECT         |       1.29 |
| EWMA (MLE lambda) | 95%     |          219 |      201.5 | 5.43%            |                19 |    0.212   | 0.043   | 0.059   | -               | -              | -               | -              |       1.15 |
| EWMA (MLE lambda) | 99%     |           95 |       40.3 | 2.36%            |                 5 |    1.7e-13 | 0.102   | 4.2e-13 | REJECT          | REJECT         | REJECT          | REJECT         |       1.16 |
| GARCH-Normal      | 95%     |          231 |      201.5 | 5.73%            |                13 |    0.037   | 0.943   | 0.112   | REJECT          | -              | -               | -              |       1.14 |
| GARCH-Normal      | 99%     |           91 |       40.3 | 2.26%            |                 4 |    5.5e-12 | 0.217   | 2.2e-11 | REJECT          | REJECT         | REJECT          | REJECT         |       1.14 |
| GARCH-t           | 95%     |          254 |      201.5 | 6.30%            |                18 |    0.00026 | 0.602   | 0.001   | REJECT          | REJECT         | REJECT          | REJECT         |       1.04 |
| GARCH-t           | 99%     |           73 |       40.3 | 1.81%            |                 4 |    3.3e-06 | 0.054   | 3.2e-06 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |

### Sens: W=500, refit daily
| model        | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |
|:-------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|
| GARCH-Normal | 95%     |          235 |      201.5 | 5.83%            |                14 |    0.018   |   0.933 | 0.061   | REJECT          | -              | -               | -              |       1.14 |
| GARCH-Normal | 99%     |           94 |       40.3 | 2.33%            |                 4 |    4.2e-13 |   0.26  | 2e-12   | REJECT          | REJECT         | REJECT          | REJECT         |       1.14 |
| GARCH-t      | 95%     |          258 |      201.5 | 6.40%            |                20 |    8.8e-05 |   0.374 | 0.0003  | REJECT          | REJECT         | REJECT          | REJECT         |       1.04 |
| GARCH-t      | 99%     |           74 |       40.3 | 1.84%            |                 4 |    1.8e-06 |   0.06  | 1.9e-06 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |

### Sens: W=250, refit 21d
| model             | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |
|:------------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|
| Historical        | 95%     |          222 |      201.5 | 5.51%            |                21 |    0.145   | 0.015   | 0.018   | -               | REJECT         | -               | -              |       1.04 |
| Historical        | 99%     |           54 |       40.3 | 1.34%            |                 6 |    0.039   | 6.6e-05 | 4.2e-05 | REJECT          | REJECT         | -               | REJECT         |       1.17 |
| Normal            | 95%     |          229 |      201.5 | 5.68%            |                25 |    0.052   | 0.002   | 0.00097 | -               | REJECT         | -               | REJECT         |       1.22 |
| Normal            | 99%     |           94 |       40.3 | 2.33%            |                 9 |    4.2e-13 | 0.00034 | 6.2e-15 | REJECT          | REJECT         | REJECT          | REJECT         |       1.28 |
| EWMA (MLE lambda) | 95%     |          227 |      201.5 | 5.63%            |                19 |    0.071   | 0.084   | 0.043   | -               | REJECT         | -               | -              |       1.16 |
| EWMA (MLE lambda) | 99%     |           96 |       40.3 | 2.38%            |                 5 |    7.1e-14 | 0.11    | 1.9e-13 | REJECT          | REJECT         | REJECT          | REJECT         |       1.18 |
| GARCH-Normal      | 95%     |          245 |      201.5 | 6.08%            |                18 |    0.002   | 0.406   | 0.007   | REJECT          | REJECT         | REJECT          | REJECT         |       1.18 |
| GARCH-Normal      | 99%     |          104 |       40.3 | 2.58%            |                 5 |    3.9e-17 | 0.193   | 1.8e-16 | REJECT          | REJECT         | REJECT          | REJECT         |       1.19 |
| GARCH-t           | 95%     |          259 |      201.5 | 6.43%            |                19 |    6.7e-05 | 0.546   | 0.00029 | REJECT          | REJECT         | REJECT          | REJECT         |       1.05 |
| GARCH-t           | 99%     |           79 |       40.3 | 1.96%            |                 4 |    6.1e-08 | 0.092   | 1e-07   | REJECT          | REJECT         | REJECT          | REJECT         |       1.03 |

### Sens: W=1000, refit 21d
| model             | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |
|:------------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|
| Historical        | 95%     |          187 |      201.5 | 4.64%            |                17 |    0.289   | 0.008   | 0.017   | -               | REJECT         | -               | -              |       1    |
| Historical        | 99%     |           42 |       40.3 | 1.04%            |                 5 |    0.789   | 5.5e-05 | 0.00028 | -               | REJECT         | -               | REJECT         |       1.04 |
| Normal            | 95%     |          178 |      201.5 | 4.42%            |                19 |    0.083   | 0.00035 | 0.00038 | -               | REJECT         | -               | REJECT         |       1.23 |
| Normal            | 99%     |           77 |       40.3 | 1.91%            |                 8 |    2.4e-07 | 9.6e-05 | 8.1e-10 | REJECT          | REJECT         | REJECT          | REJECT         |       1.29 |
| EWMA (MLE lambda) | 95%     |          228 |      201.5 | 5.66%            |                18 |    0.06    | 0.153   | 0.061   | -               | -              | -               | -              |       1.14 |
| EWMA (MLE lambda) | 99%     |           92 |       40.3 | 2.28%            |                 5 |    2.4e-12 | 0.08    | 4.5e-12 | REJECT          | REJECT         | REJECT          | REJECT         |       1.17 |
| GARCH-Normal      | 95%     |          226 |      201.5 | 5.61%            |                15 |    0.082   | 0.5     | 0.175   | -               | -              | -               | -              |       1.13 |
| GARCH-Normal      | 99%     |           86 |       40.3 | 2.13%            |                 4 |    3.3e-10 | 0.156   | 9.5e-10 | REJECT          | REJECT         | REJECT          | REJECT         |       1.13 |
| GARCH-t           | 95%     |          247 |      201.5 | 6.13%            |                18 |    0.001   | 0.446   | 0.005   | REJECT          | REJECT         | REJECT          | REJECT         |       1.03 |
| GARCH-t           | 99%     |           68 |       40.3 | 1.69%            |                 4 |    6.5e-05 | 0.033   | 3.6e-05 | REJECT          | REJECT         | REJECT          | REJECT         |       0.99 |

## Primary set-up: 99% VaR violations by year
|   year |   days |   expected@99% |   Historical |   Normal |   EWMA (MLE lambda) |   GARCH-Normal |   GARCH-t |
|-------:|-------:|---------------:|-------------:|---------:|--------------------:|---------------:|----------:|
|   2009 |      6 |            0.1 |            0 |        0 |                   0 |              0 |         0 |
|   2010 |    252 |            2.5 |            0 |        0 |                   7 |              5 |         5 |
|   2011 |    252 |            2.5 |            5 |        7 |                   7 |              7 |         6 |
|   2012 |    250 |            2.5 |            0 |        1 |                   5 |              3 |         2 |
|   2013 |    252 |            2.5 |            2 |        2 |                   5 |              5 |         5 |
|   2014 |    252 |            2.5 |            5 |        5 |                   5 |              4 |         3 |
|   2015 |    252 |            2.5 |            7 |        9 |                   6 |              8 |         7 |
|   2016 |    252 |            2.5 |            4 |        5 |                   3 |              5 |         3 |
|   2017 |    251 |            2.5 |            0 |        0 |                   6 |              6 |         4 |
|   2018 |    251 |            2.5 |           10 |       22 |                   7 |              9 |         8 |
|   2019 |    252 |            2.5 |            0 |        3 |                   5 |              3 |         2 |
|   2020 |    253 |            2.5 |            9 |       12 |                  11 |             10 |         8 |
|   2021 |    252 |            2.5 |            0 |        1 |                   5 |              5 |         3 |
|   2022 |    251 |            2.5 |            5 |       12 |                   9 |              7 |         4 |
|   2023 |    250 |            2.5 |            0 |        0 |                   2 |              1 |         1 |
|   2024 |    252 |            2.5 |            2 |        3 |                   7 |              7 |         7 |
|   2025 |    250 |            2.5 |            5 |        6 |                   5 |              6 |         5 |

## Primary GARCH-t parameter path
- nu: median 6.5, range 4.1 to 48.9
- persistence alpha+beta: median 0.9792; share of days with >= 0.999: 6.8%
- primary EWMA lambda: median 0.9199, range 0.8461 to 0.9995

## Notes
- Daily-refit sensitivity covers GARCH only (Historical/Normal/EWMA have no refit choice beyond lambda).
- Average 99% VaR of the primary GARCH-t: $17,391; Historical: $21,539.
