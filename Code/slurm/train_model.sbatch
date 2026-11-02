#!/bin/bash
#SBATCH --job-name=c2_train
#SBATCH --exclusive
#SBATCH --exclude=mscluster128,mscluster112
#SBATCH --output=%x-%j.out
set -uo pipefail
source "$PIPE/config.sh"
MODEL="${MODEL:?MODEL not set}"
VARIANT="${VARIANT:?VARIANT not set}"
DATASETS_TO_RUN="${DATASETS_TO_RUN:-$DATASET_NAMES}"
SMOKE="${SMOKE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SCRIPT="$PIPE/models/$MODEL/train.py"
[ -f "$SCRIPT" ] || { echo "unknown model $MODEL"; exit 1; }

CPUS="${SLURM_CPUS_ON_NODE:-16}"
RES="$PIPE/models/$MODEL/results"
[ "$SMOKE" = "1" ] && RES="$PIPE/models/$MODEL/results_smoke"

for NAME in $DATASETS_TO_RUN; do
    mkdir -p "$RES/${NAME}_${VARIANT}/figures" 2>/dev/null || true
done
sleep 1
echo "[$(date +%T)] job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-?}: $MODEL / $VARIANT / $DATASETS_TO_RUN (smoke=$SMOKE)"

names=()
pids=()
for NAME in $DATASETS_TO_RUN; do
    RUN="$RES/${NAME}_${VARIANT}"
    DATA="$DATASETS/${NAME}_dataset.csv"
    if [ -f "$RUN/DONE" ]; then
        echo "[$(date +%T)] $NAME/$VARIANT already complete"
        continue
    fi
    if [ ! -f "$DATA" ]; then
        echo "[$(date +%T)] missing $DATA"
        echo "missing dataset $DATA" > "$RUN/FAILED"
        mark_failed train "$MODEL/${NAME}_${VARIANT}"
        continue
    fi
    rm -f "$RUN/FAILED"
    date +%FT%T > "$RUN/STARTED"
    FLAGS=(--data "$DATA" --run-dir "$RUN" --name "$NAME" --variant "$VARIANT" --max-hours "$TRAIN_MAX_HOURS")
    [ "$SMOKE" = "1" ] && FLAGS+=(--smoke)
    echo "[$(date +%T)] launching $NAME -> $RUN"
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem=0 \
        "$PY" "$SCRIPT" "${FLAGS[@]}" $EXTRA_ARGS >> "$RUN/train.log" 2>&1 &
    pids+=($!)
    names+=("$NAME")
done

fail=0
for i in "${!pids[@]}"; do
    RUN="$RES/${names[$i]}_${VARIANT}"
    if wait "${pids[$i]}"; then
        echo "[$(date +%T)] ${names[$i]} finished"
    else
        rc=$?
        echo "[$(date +%T)] ${names[$i]} FAILED (exit $rc)"
        echo "exit $rc at $(date +%FT%T), job ${SLURM_JOB_ID:-?}" > "$RUN/FAILED"
        mark_failed train "$MODEL/${names[$i]}_${VARIANT}"
        fail=1
    fi
done
exit $fail
