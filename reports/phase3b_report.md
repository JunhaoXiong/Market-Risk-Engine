# Phase 3b Report: tail-model extensions to the rolling VaR backtest

Out-of-sample forecast dates: 2009-12-23 to 2025-12-31 (4030 days), identical to Phase 3.
Each forecast for day t uses only returns before t. New models in **bold**.
Tests: Kupiec POF, Christoffersen independence and conditional coverage; "REJECT" at the stated level.
`es_ratio` = mean realised loss on violation days / mean forecast ES on those days (descriptive, ~1 good).
`avg VaR $` = average forecast VaR on the $1,000,000 portfolio.

### Primary: W=500, refit 21d
| model                    | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |   avg VaR $ |
|:-------------------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|------------:|
| Historical               | 95%     |          204 |      201.5 | 5.06%            |                19 |    0.857   | 0.01    | 0.036   | -               | REJECT         | -               | -              |       1.02 |      12,571 |
| Historical               | 99%     |           54 |       40.3 | 1.34%            |                 6 |    0.039   | 6.6e-05 | 4.2e-05 | REJECT          | REJECT         | -               | REJECT         |       1.07 |      21,539 |
| Normal                   | 95%     |          204 |      201.5 | 5.06%            |                22 |    0.857   | 0.00073 | 0.003   | -               | REJECT         | -               | REJECT         |       1.24 |      12,692 |
| Normal                   | 99%     |           88 |       40.3 | 2.18%            |                 9 |    6.6e-11 | 0.00012 | 3.3e-13 | REJECT          | REJECT         | REJECT          | REJECT         |       1.29 |      18,163 |
| EWMA (MLE lambda)        | 95%     |          219 |      201.5 | 5.43%            |                19 |    0.212   | 0.043   | 0.059   | -               | -              | -               | -              |       1.15 |      11,529 |
| EWMA (MLE lambda)        | 99%     |           95 |       40.3 | 2.36%            |                 5 |    1.7e-13 | 0.102   | 4.2e-13 | REJECT          | REJECT         | REJECT          | REJECT         |       1.16 |      16,305 |
| GARCH-Normal             | 95%     |          231 |      201.5 | 5.73%            |                13 |    0.037   | 0.943   | 0.112   | REJECT          | -              | -               | -              |       1.14 |      11,013 |
| GARCH-Normal             | 99%     |           91 |       40.3 | 2.26%            |                 4 |    5.5e-12 | 0.217   | 2.2e-11 | REJECT          | REJECT         | REJECT          | REJECT         |       1.14 |      15,880 |
| GARCH-t                  | 95%     |          254 |      201.5 | 6.30%            |                18 |    0.00026 | 0.602   | 0.001   | REJECT          | REJECT         | REJECT          | REJECT         |       1.04 |      10,677 |
| GARCH-t                  | 99%     |           73 |       40.3 | 1.81%            |                 4 |    3.3e-06 | 0.054   | 3.2e-06 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |      17,391 |
| **GARCH-skew-t**         | 95%     |          215 |      201.5 | 5.33%            |                11 |    0.334   | 0.882   | 0.618   | -               | -              | -               | -              |       1.02 |      11,364 |
| **GARCH-skew-t**         | 99%     |           60 |       40.3 | 1.49%            |                 4 |    0.004   | 0.013   | 0.00068 | REJECT          | REJECT         | REJECT          | REJECT         |       1    |      18,674 |
| **GJR-t**                | 95%     |          244 |      201.5 | 6.05%            |                16 |    0.003   | 0.738   | 0.011   | REJECT          | REJECT         | REJECT          | -              |       1.05 |      10,752 |
| **GJR-t**                | 99%     |           72 |       40.3 | 1.79%            |                 3 |    6.2e-06 | 0.188   | 1.5e-05 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |      17,188 |
| **GJR-skew-t**           | 95%     |          205 |      201.5 | 5.09%            |                12 |    0.801   | 0.616   | 0.854   | -               | -              | -               | -              |       1.02 |      11,648 |
| **GJR-skew-t**           | 99%     |           53 |       40.3 | 1.32%            |                 3 |    0.055   | 0.037   | 0.018   | -               | REJECT         | -               | -              |       0.99 |      18,889 |
| **EGARCH-t**             | 95%     |          256 |      201.5 | 6.35%            |                18 |    0.00015 | 0.651   | 0.00068 | REJECT          | REJECT         | REJECT          | REJECT         |       1.07 |      10,653 |
| **EGARCH-t**             | 99%     |           83 |       40.3 | 2.06%            |                 4 |    3.3e-09 | 0.125   | 7.7e-09 | REJECT          | REJECT         | REJECT          | REJECT         |       1.02 |      16,896 |
| **FHS (GARCH-t filter)** | 95%     |          201 |      201.5 | 4.99%            |                 8 |    0.971   | 0.486   | 0.784   | -               | -              | -               | -              |       1.01 |      11,577 |
| **FHS (GARCH-t filter)** | 99%     |           55 |       40.3 | 1.36%            |                 4 |    0.027   | 0.007   | 0.002   | REJECT          | REJECT         | -               | REJECT         |       1.01 |      19,207 |
| **FHS (GJR-t filter)**   | 95%     |          208 |      201.5 | 5.16%            |                13 |    0.64    | 0.48    | 0.697   | -               | -              | -               | -              |       0.99 |      11,635 |
| **FHS (GJR-t filter)**   | 99%     |           53 |       40.3 | 1.32%            |                 3 |    0.055   | 0.037   | 0.018   | -               | REJECT         | -               | -              |       1    |      19,249 |

### Sens: W=500, refit daily
| model                    | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |   avg VaR $ |
|:-------------------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|------------:|
| GARCH-Normal             | 95%     |          235 |      201.5 | 5.83%            |                14 |    0.018   |   0.933 | 0.061   | REJECT          | -              | -               | -              |       1.14 |      11,066 |
| GARCH-Normal             | 99%     |           94 |       40.3 | 2.33%            |                 4 |    4.2e-13 |   0.26  | 2e-12   | REJECT          | REJECT         | REJECT          | REJECT         |       1.14 |      15,957 |
| GARCH-t                  | 95%     |          258 |      201.5 | 6.40%            |                20 |    8.8e-05 |   0.374 | 0.0003  | REJECT          | REJECT         | REJECT          | REJECT         |       1.04 |      10,704 |
| GARCH-t                  | 99%     |           74 |       40.3 | 1.84%            |                 4 |    1.8e-06 |   0.06  | 1.9e-06 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |      17,435 |
| **GARCH-skew-t**         | 95%     |          217 |      201.5 | 5.38%            |                10 |    0.268   |   0.594 | 0.468   | -               | -              | -               | -              |       1.02 |      11,394 |
| **GARCH-skew-t**         | 99%     |           59 |       40.3 | 1.46%            |                 3 |    0.006   |   0.067 | 0.004   | REJECT          | REJECT         | REJECT          | REJECT         |       1    |      18,724 |
| **GJR-t**                | 95%     |          248 |      201.5 | 6.15%            |                16 |    0.001   |   0.842 | 0.005   | REJECT          | REJECT         | REJECT          | REJECT         |       1.05 |      10,773 |
| **GJR-t**                | 99%     |           76 |       40.3 | 1.89%            |                 3 |    4.8e-07 |   0.243 | 1.6e-06 | REJECT          | REJECT         | REJECT          | REJECT         |       1    |      17,220 |
| **GJR-skew-t**           | 95%     |          205 |      201.5 | 5.09%            |                14 |    0.801   |   0.266 | 0.521   | -               | -              | -               | -              |       1.02 |      11,677 |
| **GJR-skew-t**           | 99%     |           54 |       40.3 | 1.34%            |                 3 |    0.039   |   0.041 | 0.015   | REJECT          | REJECT         | -               | -              |       0.99 |      18,935 |
| **EGARCH-t**             | 95%     |          254 |      201.5 | 6.30%            |                17 |    0.00026 |   0.794 | 0.001   | REJECT          | REJECT         | REJECT          | REJECT         |       1.06 |      10,721 |
| **EGARCH-t**             | 99%     |           83 |       40.3 | 2.06%            |                 4 |    3.3e-09 |   0.125 | 7.7e-09 | REJECT          | REJECT         | REJECT          | REJECT         |       1.02 |      16,992 |
| **FHS (GARCH-t filter)** | 95%     |          208 |      201.5 | 5.16%            |                10 |    0.64    |   0.81  | 0.87    | -               | -              | -               | -              |       1    |      11,589 |
| **FHS (GARCH-t filter)** | 99%     |           56 |       40.3 | 1.39%            |                 3 |    0.019   |   0.05  | 0.009   | REJECT          | REJECT         | -               | REJECT         |       1.02 |      19,202 |
| **FHS (GJR-t filter)**   | 95%     |          206 |      201.5 | 5.11%            |                12 |    0.746   |   0.64  | 0.85    | -               | -              | -               | -              |       1    |      11,645 |
| **FHS (GJR-t filter)**   | 99%     |           54 |       40.3 | 1.34%            |                 2 |    0.039   |   0.209 | 0.054   | REJECT          | -              | -               | -              |       1.01 |      19,252 |

### Sens: W=250, refit 21d
| model                    | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |   avg VaR $ |
|:-------------------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|------------:|
| Historical               | 95%     |          222 |      201.5 | 5.51%            |                21 |    0.145   | 0.015   | 0.018   | -               | REJECT         | -               | -              |       1.04 |      11,841 |
| Historical               | 99%     |           54 |       40.3 | 1.34%            |                 6 |    0.039   | 6.6e-05 | 4.2e-05 | REJECT          | REJECT         | -               | REJECT         |       1.17 |      20,466 |
| Normal                   | 95%     |          229 |      201.5 | 5.68%            |                25 |    0.052   | 0.002   | 0.00097 | -               | REJECT         | -               | REJECT         |       1.22 |      11,965 |
| Normal                   | 99%     |           94 |       40.3 | 2.33%            |                 9 |    4.2e-13 | 0.00034 | 6.2e-15 | REJECT          | REJECT         | REJECT          | REJECT         |       1.28 |      17,147 |
| EWMA (MLE lambda)        | 95%     |          227 |      201.5 | 5.63%            |                19 |    0.071   | 0.084   | 0.043   | -               | REJECT         | -               | -              |       1.16 |      11,410 |
| EWMA (MLE lambda)        | 99%     |           96 |       40.3 | 2.38%            |                 5 |    7.1e-14 | 0.11    | 1.9e-13 | REJECT          | REJECT         | REJECT          | REJECT         |       1.18 |      16,137 |
| GARCH-Normal             | 95%     |          245 |      201.5 | 6.08%            |                18 |    0.002   | 0.406   | 0.007   | REJECT          | REJECT         | REJECT          | REJECT         |       1.18 |      11,000 |
| GARCH-Normal             | 99%     |          104 |       40.3 | 2.58%            |                 5 |    3.9e-17 | 0.193   | 1.8e-16 | REJECT          | REJECT         | REJECT          | REJECT         |       1.19 |      15,858 |
| GARCH-t                  | 95%     |          259 |      201.5 | 6.43%            |                19 |    6.7e-05 | 0.546   | 0.00029 | REJECT          | REJECT         | REJECT          | REJECT         |       1.05 |      10,720 |
| GARCH-t                  | 99%     |           79 |       40.3 | 1.96%            |                 4 |    6.1e-08 | 0.092   | 1e-07   | REJECT          | REJECT         | REJECT          | REJECT         |       1.03 |      17,336 |
| **GARCH-skew-t**         | 95%     |          223 |      201.5 | 5.53%            |                15 |    0.126   | 0.437   | 0.228   | -               | -              | -               | -              |       1.05 |      11,443 |
| **GARCH-skew-t**         | 99%     |           65 |       40.3 | 1.61%            |                 4 |    0.00033 | 0.024   | 0.00012 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |      18,687 |
| **GJR-t**                | 95%     |          239 |      201.5 | 5.93%            |                16 |    0.008   | 0.613   | 0.027   | REJECT          | REJECT         | REJECT          | -              |       1.08 |      10,766 |
| **GJR-t**                | 99%     |           79 |       40.3 | 1.96%            |                 3 |    6.1e-08 | 0.29    | 2.4e-07 | REJECT          | REJECT         | REJECT          | REJECT         |       1.03 |      17,112 |
| **GJR-skew-t**           | 95%     |          201 |      201.5 | 4.99%            |                14 |    0.971   | 0.21    | 0.456   | -               | -              | -               | -              |       1.06 |      11,653 |
| **GJR-skew-t**           | 99%     |           63 |       40.3 | 1.56%            |                 2 |    0.0009  | 0.36    | 0.003   | REJECT          | REJECT         | REJECT          | REJECT         |       1    |      18,811 |
| **EGARCH-t**             | 95%     |          251 |      201.5 | 6.23%            |                18 |    0.00056 | 0.533   | 0.002   | REJECT          | REJECT         | REJECT          | REJECT         |       1.1  |      10,672 |
| **EGARCH-t**             | 99%     |           85 |       40.3 | 2.11%            |                 3 |    7.2e-10 | 0.4     | 3.9e-09 | REJECT          | REJECT         | REJECT          | REJECT         |       1.05 |      16,755 |
| **FHS (GARCH-t filter)** | 95%     |          212 |      201.5 | 5.26%            |                15 |    0.452   | 0.246   | 0.383   | -               | -              | -               | -              |       1.03 |      11,650 |
| **FHS (GARCH-t filter)** | 99%     |           67 |       40.3 | 1.66%            |                 4 |    0.00011 | 0.03    | 5.5e-05 | REJECT          | REJECT         | REJECT          | REJECT         |       1.02 |      18,997 |
| **FHS (GJR-t filter)**   | 95%     |          213 |      201.5 | 5.29%            |                14 |    0.41    | 0.404   | 0.502   | -               | -              | -               | -              |       1.02 |      11,559 |
| **FHS (GJR-t filter)**   | 99%     |           65 |       40.3 | 1.61%            |                 3 |    0.00033 | 0.112   | 0.00044 | REJECT          | REJECT         | REJECT          | REJECT         |       1.02 |      18,760 |

### Sens: W=1000, refit 21d
| model                    | alpha   |   violations |   expected | violation_rate   |   clustered (n11) |   kupiec_p |   ind_p |    cc_p | reject_pof@5%   | reject_cc@5%   | reject_pof@1%   | reject_cc@1%   |   es_ratio |   avg VaR $ |
|:-------------------------|:--------|-------------:|-----------:|:-----------------|------------------:|-----------:|--------:|--------:|:----------------|:---------------|:----------------|:---------------|-----------:|------------:|
| Historical               | 95%     |          187 |      201.5 | 4.64%            |                17 |    0.289   | 0.008   | 0.017   | -               | REJECT         | -               | -              |       1    |      12,944 |
| Historical               | 99%     |           42 |       40.3 | 1.04%            |                 5 |    0.789   | 5.5e-05 | 0.00028 | -               | REJECT         | -               | REJECT         |       1.04 |      23,422 |
| Normal                   | 95%     |          178 |      201.5 | 4.42%            |                19 |    0.083   | 0.00035 | 0.00038 | -               | REJECT         | -               | REJECT         |       1.23 |      13,228 |
| Normal                   | 99%     |           77 |       40.3 | 1.91%            |                 8 |    2.4e-07 | 9.6e-05 | 8.1e-10 | REJECT          | REJECT         | REJECT          | REJECT         |       1.29 |      18,905 |
| EWMA (MLE lambda)        | 95%     |          228 |      201.5 | 5.66%            |                18 |    0.06    | 0.153   | 0.061   | -               | -              | -               | -              |       1.14 |      11,589 |
| EWMA (MLE lambda)        | 99%     |           92 |       40.3 | 2.28%            |                 5 |    2.4e-12 | 0.08    | 4.5e-12 | REJECT          | REJECT         | REJECT          | REJECT         |       1.17 |      16,390 |
| GARCH-Normal             | 95%     |          226 |      201.5 | 5.61%            |                15 |    0.082   | 0.5     | 0.175   | -               | -              | -               | -              |       1.13 |      11,223 |
| GARCH-Normal             | 99%     |           86 |       40.3 | 2.13%            |                 4 |    3.3e-10 | 0.156   | 9.5e-10 | REJECT          | REJECT         | REJECT          | REJECT         |       1.13 |      16,169 |
| GARCH-t                  | 95%     |          247 |      201.5 | 6.13%            |                18 |    0.001   | 0.446   | 0.005   | REJECT          | REJECT         | REJECT          | REJECT         |       1.03 |      10,865 |
| GARCH-t                  | 99%     |           68 |       40.3 | 1.69%            |                 4 |    6.5e-05 | 0.033   | 3.6e-05 | REJECT          | REJECT         | REJECT          | REJECT         |       0.99 |      17,786 |
| **GARCH-skew-t**         | 95%     |          217 |      201.5 | 5.38%            |                15 |    0.268   | 0.324   | 0.332   | -               | -              | -               | -              |       1    |      11,539 |
| **GARCH-skew-t**         | 99%     |           50 |       40.3 | 1.24%            |                 4 |    0.139   | 0.003   | 0.004   | -               | REJECT         | -               | REJECT         |       1.01 |      19,055 |
| **GJR-t**                | 95%     |          238 |      201.5 | 5.91%            |                15 |    0.01    | 0.792   | 0.035   | REJECT          | REJECT         | -               | -              |       1.03 |      10,936 |
| **GJR-t**                | 99%     |           68 |       40.3 | 1.69%            |                 3 |    6.5e-05 | 0.141   | 0.00012 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |      17,585 |
| **GJR-skew-t**           | 95%     |          202 |      201.5 | 5.01%            |                12 |    0.971   | 0.546   | 0.833   | -               | -              | -               | -              |       1    |      11,806 |
| **GJR-skew-t**           | 99%     |           54 |       40.3 | 1.34%            |                 2 |    0.039   | 0.209   | 0.054   | REJECT          | -              | -               | -              |       0.97 |      19,247 |
| **EGARCH-t**             | 95%     |          239 |      201.5 | 5.93%            |                16 |    0.008   | 0.613   | 0.027   | REJECT          | REJECT         | REJECT          | -              |       1.05 |      10,918 |
| **EGARCH-t**             | 99%     |           68 |       40.3 | 1.69%            |                 2 |    6.5e-05 | 0.463   | 0.00026 | REJECT          | REJECT         | REJECT          | REJECT         |       1.01 |      17,463 |
| **FHS (GARCH-t filter)** | 95%     |          192 |      201.5 | 4.76%            |                11 |    0.489   | 0.532   | 0.649   | -               | -              | -               | -              |       0.99 |      11,931 |
| **FHS (GARCH-t filter)** | 99%     |           47 |       40.3 | 1.17%            |                 4 |    0.301   | 0.002   | 0.005   | -               | REJECT         | -               | REJECT         |       1    |      19,798 |
| **FHS (GJR-t filter)**   | 95%     |          201 |      201.5 | 4.99%            |                11 |    0.971   | 0.75    | 0.95    | -               | -              | -               | -              |       0.98 |      11,891 |
| **FHS (GJR-t filter)**   | 99%     |           47 |       40.3 | 1.17%            |                 2 |    0.301   | 0.124   | 0.179   | -               | -              | -               | -              |       0.99 |      19,687 |

## Primary set-up: 99% VaR violations by year
|   year |   days |   expected@99% |   Historical |   GARCH-t |   GARCH-skew-t |   GJR-t |   GJR-skew-t |   EGARCH-t |   FHS (GARCH-t filter) |   FHS (GJR-t filter) |
|-------:|-------:|---------------:|-------------:|----------:|---------------:|--------:|-------------:|-----------:|-----------------------:|---------------------:|
|   2009 |      6 |            0.1 |            0 |         0 |              0 |       0 |            0 |          0 |                      0 |                    0 |
|   2010 |    252 |            2.5 |            0 |         5 |              2 |       3 |            1 |          4 |                      4 |                    3 |
|   2011 |    252 |            2.5 |            5 |         6 |              4 |       6 |            4 |          7 |                      4 |                    4 |
|   2012 |    250 |            2.5 |            0 |         2 |              1 |       2 |            1 |          2 |                      1 |                    2 |
|   2013 |    252 |            2.5 |            2 |         5 |              5 |       4 |            4 |          3 |                      5 |                    3 |
|   2014 |    252 |            2.5 |            5 |         3 |              3 |       4 |            2 |          5 |                      3 |                    3 |
|   2015 |    252 |            2.5 |            7 |         7 |              5 |       7 |            5 |          8 |                      5 |                    5 |
|   2016 |    252 |            2.5 |            4 |         3 |              3 |       2 |            2 |          3 |                      3 |                    2 |
|   2017 |    251 |            2.5 |            0 |         4 |              4 |       5 |            5 |          6 |                      3 |                    4 |
|   2018 |    251 |            2.5 |           10 |         8 |              7 |       8 |            6 |          8 |                      5 |                    5 |
|   2019 |    252 |            2.5 |            0 |         2 |              2 |       5 |            3 |          4 |                      2 |                    3 |
|   2020 |    253 |            2.5 |            9 |         8 |              7 |       7 |            6 |         11 |                      5 |                    7 |
|   2021 |    252 |            2.5 |            0 |         3 |              2 |       4 |            3 |          4 |                      2 |                    2 |
|   2022 |    251 |            2.5 |            5 |         4 |              2 |       6 |            1 |          3 |                      1 |                    1 |
|   2023 |    250 |            2.5 |            0 |         1 |              1 |       0 |            0 |          3 |                      1 |                    0 |
|   2024 |    252 |            2.5 |            2 |         7 |              7 |       5 |            6 |          7 |                      6 |                    5 |
|   2025 |    250 |            2.5 |            5 |         5 |              5 |       4 |            4 |          5 |                      5 |                    4 |

## Primary set-up: parameter medians over refits
| model        |   persistence (median) |   nu (median) |   skew lambda (median) |   share lambda<0 |   gamma (median) |
|:-------------|-----------------------:|--------------:|-----------------------:|-----------------:|-----------------:|
| GARCH-t      |                  0.979 |         6.546 |                nan     |          nan     |          nan     |
| GARCH-skew-t |                  0.979 |         6.906 |                 -0.103 |            0.932 |          nan     |
| GJR-t        |                  0.970 |         7.361 |                nan     |          nan     |            0.192 |
| GJR-skew-t   |                  0.973 |         7.659 |                 -0.133 |            0.943 |            0.202 |
| EGARCH-t     |                  0.959 |         7.494 |                nan     |          nan     |           -0.159 |

## Refits rejected as numerically implausible (previous parameters kept)
- Primary: W=500, refit 21d: none
- Sens: W=500, refit daily: none
- Sens: W=250, refit 21d: none
- Sens: W=1000, refit 21d: none

Run time per set-up (s): Primary: W=500, refit 21d: 62, Sens: W=500, refit daily: 1008, Sens: W=250, refit 21d: 63, Sens: W=1000, refit 21d: 64

## Notes
- Phase 3 models are recomputed here and reproduce the Phase 3 numbers exactly.
- FHS residual pool = the last W standardised residuals (W = estimation window), filtered with
  the parameters of the latest refit and rolled forward daily; W = 250 / 1000 is the pool-size sensitivity.
- EGARCH is estimated with alpha >= 0 and beta <= 0.999 and from four starting points (best plausible
  likelihood kept): the unrestricted optimum on some 500-day windows had alpha < 0, beta ~ 1 and
  exploded out of sample. A refit with no plausible optimum would keep the previous parameters.
- Six new candidates were evaluated on the same sample; p-values are not adjusted for this.
