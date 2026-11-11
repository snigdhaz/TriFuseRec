#!/usr/bin/env bash
# Amazon Beauty: preprocessing, MostPopular, DeepFM / SASRec / NGCF x seeds, then the same
# gating comparison as ML-1M (src/cand_gate.py: best single, equal, fixed, per-user gate,
# candidate-aware gate) and the report (src/report.py).
#
#   PY=python SEEDS="42 43 44" bash scripts/run_beauty.sh
#
# Resumable: finished training runs (results/beauty/<model>_seed<k>.csv) and gate runs
# (results/beauty/cand_gate/seed<k>/metrics.csv) are skipped. If a run's score matrices are
# missing they are regenerated from the saved .pt with --eval-only.
# Score matrices are ~1 GB each (6.5 GB per seed). CLEAN_SCORES=1 deletes a seed's scores after
# its gate run (use on Colab / small disks). SCORES_DIR puts them elsewhere (e.g. local disk).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
DATA=data/processed_beauty; OUT=results/beauty; LOG=$OUT/logs; mkdir -p "$LOG"
SCORES=${SCORES_DIR:-$OUT/scores}; mkdir -p "$SCORES"

[ -f $DATA/train.csv ] || $PY -m src.preprocess_beauty --tie-break ${TIE_BREAK:-random}
[ -f $OUT/popularity.csv ] || $PY -m src.popularity --data $DATA --out $OUT | tee $LOG/popularity.log

train() {  # train <model> <seed> <extra args...>
  local model=$1 seed=$2; shift 2
  local args=(--dataset beauty --model $model --seed $seed --dim 32 --scores-dir $SCORES "$@")
  if [ -f $OUT/${model}_seed${seed}.csv ]; then
    if [ -f $SCORES/${model}_seed${seed}_test.npy ] || [ -f $OUT/cand_gate/seed${seed}/metrics.csv ]; then
      echo "skip $model seed $seed (done)"; return; fi
    echo "re-scoring $model seed $seed from saved weights"; args+=(--eval-only)
  fi
  $PY -u -m src.train "${args[@]}" 2>&1 | { grep -v -i "warning\|adj=torch" || true; } | tee $LOG/${model}_seed${seed}.log
}

for seed in ${SEEDS:-42 43 44}; do
  train sasrec $seed --epochs 40 --eval-every 2 --patience 3
  train ngcf   $seed --epochs 100 --eval-every 5 --patience 3
  train deepfm $seed --init-std 0.01 --epochs 20 --eval-every 2 --patience 2
  if [ ! -f $OUT/cand_gate/seed$seed/metrics.csv ]; then
    $PY -u -m src.cand_gate --data $DATA --scores $SCORES --score-seed $seed --seed $seed \
        --out $OUT/cand_gate/seed$seed --overwrite 2>&1 | { grep -v -i "warning" || true; } | tee $LOG/cand_gate_seed$seed.log
  fi
  if [ "${CLEAN_SCORES:-0}" = 1 ]; then rm -f $SCORES/*_seed${seed}_*.npy; fi
done

$PY -m src.compare --out $OUT --seed 42 | tee $LOG/compare.log
for split in validation test; do $PY -m src.analyze --dataset beauty --split $split | tee $LOG/analyze_$split.log; done
$PY -m src.report --runs $OUT/cand_gate/seed* --tag beauty --data $DATA --out-root $OUT/analysis \
    --timestamp-comparison '' | tee $LOG/report.log
echo ALL_DONE
