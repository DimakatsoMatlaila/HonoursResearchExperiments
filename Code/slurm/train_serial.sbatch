#!/bin/bash
#SBATCH --job-name=c2_serial
#SBATCH --exclusive
#SBATCH --exclude=mscluster128,mscluster112
#SBATCH --output=%x-%j.out
set -uo pipefail
source "$PIPE/config.sh"

START=$(date +%s)
LIMIT_H="${LIMIT_H:-72}"
MODELS_TO_RUN="${MODELS_TO_RUN:-$MODELS}"
VARIANTS_TO_RUN="${VARIANTS_TO_RUN:-$VARIANTS}"
DATASETS_TO_RUN="${DATASETS_TO_RUN:-master external internal}"
CPUS="${SLURM_CPUS_ON_NODE:-14}"

echo "[$(date +%T)] serial job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-?} ($CPUS cpus)"

if [ -n "${WAIT_FOR_JOB:-}" ]; then
    echo "[$(date +%T)] holding this node until job $WAIT_FOR_JOB leaves the queue"
    while squeue -h -j "$WAIT_FOR_JOB" 2>/dev/null | grep -q .; do sleep 60; done
    for i in 1 2 3 4 5 6 7 8 9 10; do
        ok=1
        for d in $DATASETS_TO_RUN; do [ -f "$DATASETS/${d}_dataset.csv" ] || ok=0; done
        [ "$ok" = 1 ] && break
        sleep 60
    done
fi

for d in $DATASETS_TO_RUN; do
    [ -f "$DATASETS/${d}_dataset.csv" ] || { echo "missing $DATASETS/${d}_dataset.csv - build did not finish?"; exit 1; }
done

for m in $MODELS_TO_RUN; do
    for v in $VARIANTS_TO_RUN; do
        for d in $DATASETS_TO_RUN; do
            run="$PIPE/models/$m/results_serial/${d}_${v}"
            if [ -f "$run/DONE" ]; then echo "[$(date +%T)] $m/${d}_$v already complete"; continue; fi
            hours=$(awk -v l="$LIMIT_H" -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.1f", l-(n-s)/3600-1}')
            if awk -v h="$hours" 'BEGIN{exit !(h<1)}'; then
                echo "[$(date +%T)] out of time; resubmit to continue"
                exit 0
            fi
            mkdir -p "$run/figures" 2>/dev/null || true
            rm -f "$run/FAILED"
            date +%FT%T > "$run/STARTED"
            echo "[$(date +%T)] start  $m/${d}_$v (budget ${hours}h)"
            if srun --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem=0 \
                    "$PY" "$PIPE/models/$m/train.py" --data "$DATASETS/${d}_dataset.csv" --run-dir "$run" \
                    --name "$d" --variant "$v" --max-hours "$hours" ${EXTRA_ARGS:-} >> "$run/train.log" 2>&1; then
                echo "[$(date +%T)] done   $m/${d}_$v"
            else
                rc=$?
                echo "exit $rc at $(date +%FT%T), job ${SLURM_JOB_ID:-?}" > "$run/FAILED"
                mark_failed train "$m/${d}_$v"
                echo "[$(date +%T)] FAILED $m/${d}_$v (exit $rc), continuing with the next"
            fi
        done
    done
done
echo "[$(date +%T)] serial run finished"
