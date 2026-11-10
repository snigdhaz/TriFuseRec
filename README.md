# TriFuseRec: Adaptive Context-Sequence-Graph Fusion for Personalized Recommendation

TriFuseRec is a personalized recommendation framework that adaptively combines three complementary sources of information: user-item features, sequential interaction history, and user-item graph structure.

The project integrates three recommendation architectures:

- **DeepFM:** Captures user and item feature interactions.
- **SASRec:** Models sequential user preferences using self-attention.
- **NGCF:** Learns collaborative relationships through graph-based representation learning.

Unlike traditional recommendation systems that rely on a single model or fixed ensemble weights, TriFuseRec introduces a personalized gating network that learns how much to rely on each model based on user-specific characteristics, such as interaction history length and item popularity.

## Dataset
MovieLens-1M, containing approximately one million user-movie interactions.

## Evaluation
Models are evaluated using Recall@10, Recall@20, NDCG@10, and NDCG@20, along with user-group performance, recommendation diversity, and popularity bias.

## Team Members
- Mahek Viradiya
- Snigdha Challapalli
- Srushti Thakar

## Pipeline (single run, current loo2 split)
```bash
pip install -r requirements.txt
python -m src.preprocess --seed 42              # ML-1M -> per-user chronological leave-last-two-out
python -m src.popularity                        # MostPopular (validation + test)
# Early stopping on validation NDCG@10; add --eval-only to re-score a saved .pt (same --dim).
python -m src.train --model sasrec --dim 64 --epochs 50  --patience 3
python -m src.train --model ngcf   --dim 64 --epochs 400 --eval-every 5 --patience 4
python -m src.train --model deepfm --dim 64 --epochs 30  --neg 4 --patience 3
python -m src.fusion                            # original per-user gate + fixed-weight ensembles
python -m src.cand_gate --out results/cand_gate/loo2_seed42   # per-user vs candidate-aware gate
python -m src.report --runs results/cand_gate/loo2_seed42 --tag loo2_seed42
python -m pytest -q
```

## Multi-seed experiments (recommended protocol)
`loo3` holds out one more item per user, so the gate is fit on one target (fusion),
selected on another (validation) and reported on a third (test). The preprocessing
seed is fixed, so all training seeds share identical test targets and can be paired.
```bash
python -m src.runner estimate                                   # time / disk estimate
python -m src.runner run configs/main_loo3.json --dry-run       # show the plan
python -m src.runner run configs/main_loo3.json --seeds 42      # one seed (~45-55 min on an M-series Mac)
python -m src.runner run configs/main_loo3.json                 # all 5 seeds (resumes; never overwrites)
python -m src.runner aggregate configs/main_loo3.json           # mean ± std, paired bootstrap CIs
python -m src.runner run configs/ablations_loo3.json --stages ablation
python -m src.runner ablations configs/ablations_loo3.json
python -m src.report --runs runs/main_loo3/seed*/fusion --tag main_loo3 \
    --ablations runs/main_loo3/aggregate/ablation_summary.csv
```

## Code map
| File | Purpose |
|---|---|
| `src/preprocess.py` | Versioned split (`v2`, `--protocol loo2/loo3`) |
| `src/common.py` | Data loading, `heldout_splits` (one history definition for every model), ranking metrics |
| `src/train.py`, `src/models.py` | DeepFM, SASRec, NGCF training + full-catalog score matrices |
| `src/fusion.py` | Original per-user gate (`--recency history` default; `target_time` = legacy) |
| `src/cand_gate.py` | Controlled comparison: best single, equal, fixed, per-user gate, candidate-aware gate |
| `src/runner.py`, `configs/` | Multi-seed runner, manifests, resume, aggregation, ablations |
| `src/stats.py` | Per-user metrics, paired bootstrap |
| `src/compare_runs.py` | Before/after timestamp-fix comparison |
| `src/report.py` | Research tables and figures (RQ1-RQ4) |
| `docs/AUDIT.md` | Evaluation-protocol audit, findings, versioning |

## Project Status
In development.
- [x] Preprocessing, MostPopular, DeepFM / SASRec / NGCF baselines (validation + test)
- [x] Timestamp tie-break fix (v2) and before/after comparison (`results/analysis/`)
- [x] Evaluation audit (`docs/AUDIT.md`) with regression tests
- [x] Per-user and candidate-aware gating, user-group / coverage / long-tail analysis
- [x] Multi-seed runner, paired bootstrap, ablation configs
- [ ] Run `configs/main_loo3.json` for 5 seeds and the ablations
- [ ] Inference API (after the loo3 results are in)
