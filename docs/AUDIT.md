# Evaluation protocol audit (MovieLens-1M)

Audit date: 2026-11-08. Scope: `src/preprocess.py`, `src/common.py`, `src/train.py`,
`src/models.py`, `src/fusion.py`, `src/popularity.py`, `src/analysis.py`.
Regression tests: `tests/test_protocol.py`, `tests/test_cand_gate.py`, `tests/test_models.py`.

## Protocol versions

| Version | What changed | How to reproduce |
|---|---|---|
| v1 (git ≤ `b6701e3`) | Ratings with the same timestamp ordered by item id | `git show b6701e3:data/processed/...`; `read_data` falls back to the v1 order when there is no `seq` column |
| v2 (current `data/processed`) | Ties broken randomly with `--seed` (default 42); `seq` column | `python -m src.preprocess --seed 42` |
| v2 + `loo3` (new, recommended for fusion) | Extra held-out item per user (`fusion.csv`, 3rd-to-last) to fit gates on | `python -m src.preprocess --protocol loo3 --out data/splits/loo3_tie42` |

`stats.json` records `preprocess_version`, `protocol` and `tie_break_seed` for every new split.
The existing `data/processed/stats.json` predates the version field; it is the v2 + loo2 split.

## Findings

| # | Finding | Severity | Impact | Action |
|---|---|---|---|---|
| 1 | **Timestamp ties ordered by item id (v1).** 53% of consecutive events share a timestamp; for 42% of users the test item shared the validation item's timestamp and *always* had the larger id. | High (fixed before this audit) | SASRec test Recall@10 0.329 → 0.297 after the fix (−9.7%); NDCG@10 −14.5%. Other models barely moved. See `results/analysis/timestamp_fix_comparison.md`. | Fixed in v2. Test: `test_tie_break_is_not_item_id_order`. |
| 2 | **Per-user gate used the held-out target's timestamp.** `fusion.py`'s `recency_days` was *target timestamp − last history timestamp*. That reveals whether the target was rated in the same session as the previous movie, exactly when SASRec is strong (SASRec test Recall@10 0.406 when the gap is 0 vs 0.051 when > 1 day). It was the gate's strongest input (corr. −0.80 with the SASRec weight). | High | Archived gate results (`results/fusion_seed42.csv`, `trifuserec_gate` rows) used this feature. Re-run with the leak-free feature (seed 42): test Recall@10 0.3053 → 0.3066, NDCG@10 0.1761 → 0.1762, so accuracy did not depend on it. The gate's behaviour did: the recency–SASRec-weight correlation fell from −0.80 to −0.42, so "trusts SASRec less after long gaps" was largely driven by target metadata. | `fusion.py --recency history` (gap between the last two **history** events) is now the default; `--recency target_time` reproduces the old results and is recorded in the output CSV. `cand_gate.py` never uses held-out timestamps. Test: `test_history_features_ignore_heldout_timestamps`. |
| 3 | **Gate fit and selected on the same labels.** `fusion.py` fits the gate on validation targets (80% of users) and early-stops on the other 20%; the fixed weights are grid-searched on validation of all users; the base models were also early-stopped on validation of all users. | Medium | Validation/selection numbers are optimistic; **test numbers are not affected** (test labels were never used for fitting or selection). | New `loo3` protocol: gate fits on the fusion item, selects on validation, reports on test. Under loo2, `cand_gate.py` uses disjoint fit/select **users** and documents the remaining base-model caveat. |
| 4 | **Asymmetric test-time history.** At test time SASRec reads the validation item; NGCF's propagation graph does not contain the validation edge; DeepFM uses no history. | Low | Measured: NGCF test Recall@10 0.0806 → 0.0811 when the validation edge is added. Negligible, but a fairness gap in principle. | `train.py --ngcf-infer-graph history` adds held-out history edges at inference (default `train` keeps v2 results). Enabled in `configs/main_loo3.json`. |
| 5 | **Per-user, not global, chronological split.** Other users' training interactions can be later than a user's test event: for the median user, 44% of all training interactions occur after their test timestamp. This affects item popularity, the NGCF graph, DeepFM's item embeddings, and gate popularity features. | Medium (limitation) | Standard for leave-last-out ML-1M work, but results should not be described as free of temporal leakage. | Documented here and in `preprocess.py`. A global time split is out of scope for this protocol version. |
| 6 | **Negative sampling excludes train positives only.** DeepFM/NGCF negatives may (rarely, ~2/3416 per draw) be a user's held-out item. | None (correct) | Excluding held-out items would leak them; the current behaviour adds slight label noise that can only hurt the model. | Kept. Test: `test_negative_sampling_never_returns_a_known_positive`. |
| 7 | **Popularity baseline reported validation only.** | Low | Before/after comparison needed test numbers. | `popularity.py` now writes every held-out split; `compare_runs.py` recomputes popularity on both splits. |
| 8 | **Determinism.** Preprocessing is byte-identical for a fixed seed. Training on Apple MPS is not guaranteed bit-reproducible (non-deterministic kernels), even with fixed seeds. | Low | Seed-to-seed variance must be measured, not assumed zero. | Multi-seed runner (`src/runner.py`). Test: `test_preprocess_is_deterministic_for_a_seed`. |
| 9 | **Tied scores.** `rank_metrics` uses a stable argsort, so exact ties favour lower item ids. | Very low | Only matters for integer-valued scores (MostPopular). | Documented. |

## Checks that passed

- **Splits:** for every user, train events precede the validation event, which precedes the test event (`seq` and timestamp). Held-out items never appear in that user's training history, and validation ≠ test. (`test_processed_split_is_per_user_chronological`)
- **Histories:** the history used for each split is train + all *earlier* held-out items, built in one place (`common.heldout_splits`) and shared by every model, MostPopular and the fusion code. (`test_loo2_test_history_contains_validation_item`, `test_loo3_split_and_heldout_histories`)
- **Candidates:** every method ranks all items minus that split's history; `rank_metrics` raises if a target is masked and never returns a masked item. (`test_rank_metrics_never_recommends_masked_items`)
- **Graph:** the NGCF training graph is built from `train.csv` only.
- **Features:** DeepFM uses demographics and genres only. Gate inputs in `cand_gate.py` come from histories, train popularity, item metadata and expert scores. Score normalization uses unseen items without looking at labels, and zero-variance users map to 0 rather than NaN. (`test_normalize_handles_masks_and_zero_variance`)
- **Refactor check:** the refactored `train.py --eval-only` reproduces the corrected seed-42 SASRec and NGCF test metrics exactly.

## Not changed (would alter the protocol, needs a team decision)

- Retraining base models on train + validation before scoring test.
- A global time split.
