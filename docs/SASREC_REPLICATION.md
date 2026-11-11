# SASRec replication on MovieLens-1M

Goal: check that our data pipeline and SASRec reproduce the published result of
Kang & McAuley, *Self-Attentive Sequential Recommendation* (ICDM 2018, arXiv 1808.09781),
so that our baselines and fusion results rest on a verified implementation.

## Result (test set, 100 sampled negatives, seed 42)

| Model | Data split | HR@10 | NDCG@10 | vs paper (HR / NDCG) |
|---|---|---|---|---|
| Paper (Table III) | their release | 0.8245 | 0.5905 | |
| **Paper config, ours** | `data/processed` (v2, random tie order) | **0.8210** | **0.5930** | −0.4% / +0.4% |
| Paper config, ours | v1 (ties ordered by movie id) | 0.8139 | 0.5845 | −1.3% / −1.0% |
| Team SASRec (`main`, full-softmax CE, max_len 50) | `data/processed` (v2) | 0.8164 | 0.6136 | −1.0% / +3.9% |

Sampled metrics are averaged over 5 independent draws of the 100 negatives (std ≤ 0.003).
**Our implementation and data reproduce the paper within about 1%.**

## Dataset check

The paper's Table II reports MovieLens-1M after preprocessing as 6,040 users, 3,416 items,
163.5 actions per user. Our `data/processed` split has exactly 6,040 users, 3,416 items and
987,531 training actions (163.5 per user). Preprocessing is the same: every rating is an
implicit interaction, users and items with fewer than 5 actions are dropped, sequences are
ordered by timestamp, last item = test, second-to-last = validation.

The paper does not say how ratings with the same timestamp are ordered (53% of consecutive
ML-1M ratings tie). Both tie orders reproduce the paper; they differ by under 1%.

## Settings (paper text and the authors' reference code)

| | Value |
|---|---|
| Embedding size d | 50 |
| Blocks / heads | 2 / 1 |
| Max sequence length | 200 |
| Dropout | 0.2 |
| Optimizer | Adam, lr 0.001, betas (0.9, 0.98), batch 128, no L2 |
| Loss | binary cross-entropy, one random unseen negative per position, resampled every epoch |
| Architecture | learned positions, embeddings × √d, pre-LayerNorm blocks, shared item embedding for prediction |
| Training | 200 epochs (paper: early stop after 20 epochs without validation gain); validate every 20 epochs, keep the best validation NDCG@10 checkpoint |
| Evaluation | rank the held-out item against 100 random items the user never interacted with; Hit@10, NDCG@10 |

Validation NDCG@10 (random tie order) rose to 0.606 by epoch 100 and plateaued around
0.613 from epoch 160; the best checkpoint was epoch 200 for both runs.

Small, documented differences from the reference code: users are visited once per epoch in
random order (reference: sampled with replacement); evaluation negatives also exclude the
earlier held-out item; sampled metrics are averaged over 5 draws.

## Sampled vs full-catalog metrics

The same models ranked against **all** 3,416 movies:

| Model | Test Recall@10 (full) | Test NDCG@10 (full) |
|---|---|---|
| Paper config (BCE, 1 negative), v2 | 0.241 | 0.129 |
| Team SASRec (full-softmax CE), v2 | 0.297 | 0.167 |

Two points for the report:
1. The paper's 0.82 and our project's ~0.30 are the same kind of model measured two ways.
   Sampled-100 metrics are much higher and can rank models differently from full ranking
   (Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020), which is why
   the project reports full-catalog metrics.
2. The paper configuration ties the team model on sampled HR@10 but is clearly worse on full
   ranking (0.241 vs 0.297). Training with a full softmax instead of one sampled negative is
   known to help SASRec, so the team's model is the stronger baseline for fusion.

## Reproduce

```bash
python -m src.sasrec_paper --tag v2_random_ties                         # ~2 h on a laptop CPU
python -m src.preprocess ...  # or extract the v1 split: git show b6701e3:data/processed/<file>
python -m src.sasrec_paper --data data/splits/v1_itemid --tag v1_itemid_ties
python -m src.train --model sasrec --dim 64 --eval-only                 # writes results/scores/
python -m src.sasrec_paper --score-prefix results/scores/sasrec_seed42 --tag main_sasrec_existing
python -m pytest -q tests/test_sasrec_paper.py
```

Outputs: `results/sasrec_paper/{tag}.csv` (sampled and full metrics, validation and test),
`{tag}_log.json` (validation curve), `{tag}.pt` (best weights). Add `--resume` to continue an
interrupted run from its last validation checkpoint.
