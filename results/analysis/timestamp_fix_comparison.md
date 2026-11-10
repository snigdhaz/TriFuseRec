# Timestamp tie-break fix: before vs after (seed 42)

OLD = ties ordered by item id. NEW = ties ordered randomly (seed 42).
Held-out target changed for 38.3% of users (validation) and 26.8% (test), so OLD and NEW are **not paired** comparisons.
Popularity was recomputed on both splits (the old run only saved validation).

## Test

| Method | Recall@10 old → new (Δ, rel.) | Recall@20 old → new (Δ, rel.) | NDCG@10 old → new (Δ, rel.) | NDCG@20 old → new (Δ, rel.) |
|---|---|---|---|---|
| popularity | 0.0353 → 0.0419 (+0.0066, +18.8%) | 0.0637 → 0.0702 (+0.0065, +10.1%) | 0.0173 → 0.0207 (+0.0034, +19.4%) | 0.0244 → 0.0278 (+0.0033, +13.7%) |
| deepfm | 0.0818 → 0.0811 (-0.0007, -0.8%) | 0.1389 → 0.1440 (+0.0051, +3.7%) | 0.0389 → 0.0386 (-0.0004, -0.9%) | 0.0533 → 0.0543 (+0.0010, +1.9%) |
| sasrec | 0.3286 → 0.2967 (-0.0320, -9.7%) | 0.4373 → 0.4151 (-0.0222, -5.1%) | 0.1950 → 0.1668 (-0.0282, -14.5%) | 0.2223 → 0.1967 (-0.0256, -11.5%) |
| ngcf | 0.0737 → 0.0806 (+0.0070, +9.4%) | 0.1228 → 0.1346 (+0.0118, +9.6%) | 0.0368 → 0.0406 (+0.0038, +10.3%) | 0.0491 → 0.0541 (+0.0049, +10.0%) |
| equal_weights | 0.2197 → 0.2200 (+0.0003, +0.2%) | 0.3225 → 0.3232 (+0.0007, +0.2%) | 0.1246 → 0.1189 (-0.0057, -4.6%) | 0.1504 → 0.1447 (-0.0056, -3.8%) |
| best_fixed_weights | 0.3338 → 0.3056 (-0.0281, -8.4%) | 0.4425 → 0.4257 (-0.0169, -3.8%) | 0.1998 → 0.1752 (-0.0246, -12.3%) | 0.2274 → 0.2056 (-0.0219, -9.6%) |
| trifuserec_gate | 0.3336 → 0.3053 (-0.0283, -8.5%) | 0.4457 → 0.4270 (-0.0187, -4.2%) | 0.2000 → 0.1761 (-0.0239, -11.9%) | 0.2283 → 0.2069 (-0.0214, -9.4%) |

## Validation

| Method | Recall@10 old → new (Δ, rel.) | Recall@20 old → new (Δ, rel.) | NDCG@10 old → new (Δ, rel.) | NDCG@20 old → new (Δ, rel.) |
|---|---|---|---|---|
| popularity | 0.0417 → 0.0435 (+0.0018, +4.4%) | 0.0699 → 0.0728 (+0.0030, +4.3%) | 0.0206 → 0.0215 (+0.0009, +4.3%) | 0.0276 → 0.0288 (+0.0012, +4.3%) |
| deepfm | 0.0899 → 0.0914 (+0.0015, +1.7%) | 0.1510 → 0.1570 (+0.0060, +3.9%) | 0.0443 → 0.0452 (+0.0009, +2.1%) | 0.0596 → 0.0617 (+0.0021, +3.5%) |
| sasrec | 0.3561 → 0.3136 (-0.0425, -11.9%) | 0.4720 → 0.4379 (-0.0341, -7.2%) | 0.2053 → 0.1730 (-0.0323, -15.7%) | 0.2347 → 0.2045 (-0.0302, -12.9%) |
| ngcf | 0.0820 → 0.0887 (+0.0068, +8.3%) | 0.1389 → 0.1508 (+0.0119, +8.6%) | 0.0402 → 0.0440 (+0.0038, +9.4%) | 0.0545 → 0.0595 (+0.0051, +9.3%) |
| equal_weights | 0.2545 → 0.2412 (-0.0132, -5.2%) | 0.3672 → 0.3555 (-0.0118, -3.2%) | 0.1407 → 0.1298 (-0.0108, -7.7%) | 0.1691 → 0.1586 (-0.0105, -6.2%) |
| best_fixed_weights | 0.3719 → 0.3338 (-0.0381, -10.2%) | 0.4810 → 0.4487 (-0.0323, -6.7%) | 0.2164 → 0.1863 (-0.0301, -13.9%) | 0.2440 → 0.2153 (-0.0286, -11.7%) |
| trifuserec_gate | 0.3738 → 0.3359 (-0.0379, -10.1%) | 0.4853 → 0.4555 (-0.0298, -6.1%) | 0.2171 → 0.1874 (-0.0297, -13.7%) | 0.2452 → 0.2175 (-0.0277, -11.3%) |

## Key comparisons (test)

| Quantity | Metric | OLD | NEW |
|---|---|---|---|
| SASRec − best other single model | Recall@10 | 0.2469 | 0.2156 |
| SASRec / best other single model | Recall@10 | 4.0182 | 3.6571 |
| Gate − best fixed weights | Recall@10 | -0.0002 | -0.0003 |
| Best fixed − SASRec | Recall@10 | 0.0051 | 0.0089 |
| SASRec − best other single model | Recall@20 | 0.2983 | 0.2710 |
| SASRec / best other single model | Recall@20 | 3.1478 | 2.8816 |
| Gate − best fixed weights | Recall@20 | 0.0031 | 0.0013 |
| Best fixed − SASRec | Recall@20 | 0.0053 | 0.0106 |
| SASRec − best other single model | NDCG@10 | 0.1560 | 0.1262 |
| SASRec / best other single model | NDCG@10 | 5.0077 | 4.1067 |
| Gate − best fixed weights | NDCG@10 | 0.0002 | 0.0009 |
| Best fixed − SASRec | NDCG@10 | 0.0048 | 0.0084 |
| SASRec − best other single model | NDCG@20 | 0.1690 | 0.1424 |
| SASRec / best other single model | NDCG@20 | 4.1714 | 3.6208 |
| Gate − best fixed weights | NDCG@20 | 0.0009 | 0.0013 |
| Best fixed − SASRec | NDCG@20 | 0.0051 | 0.0088 |

## Mean per-user gate weights by cohort (test)

| Cohort type | Cohort | OLD D / S / N | NEW D / S / N |
|---|---|---|---|
| history_len | <50 | 0.13 / 0.85 / 0.02 | 0.14 / 0.82 / 0.05 |
| history_len | 50-99 | 0.16 / 0.78 / 0.05 | 0.15 / 0.76 / 0.08 |
| history_len | 100-199 | 0.19 / 0.72 / 0.08 | 0.18 / 0.71 / 0.11 |
| history_len | 200-499 | 0.23 / 0.66 / 0.10 | 0.23 / 0.65 / 0.12 |
| history_len | 500+ | 0.29 / 0.60 / 0.11 | 0.29 / 0.58 / 0.13 |
| activity | Q1 (least active) | 0.23 / 0.68 / 0.09 | 0.23 / 0.66 / 0.12 |
| activity | Q2 | 0.14 / 0.83 / 0.03 | 0.14 / 0.80 / 0.06 |
| activity | Q3 | 0.16 / 0.79 / 0.05 | 0.16 / 0.77 / 0.07 |
| activity | Q4 (most active) | 0.20 / 0.72 / 0.08 | 0.19 / 0.71 / 0.10 |
