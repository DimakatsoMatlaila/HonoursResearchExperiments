#!/bin/bash
#SBATCH --job-name=c2_bundle
#SBATCH --exclusive
#SBATCH --exclude=mscluster128,mscluster112
#SBATCH --output=%x-%j.out
set -uo pipefail
source "$PIPE/config.sh"

MODELS_TO_RUN="${MODELS_TO_RUN:-dqn_gwo_rf gwo_rf lsh_rf isolation_forest hist_gb rf_baseline}"
VARIANTS_TO_RUN="${VARIANTS_TO_RUN:-$VARIANTS}"
DATASETS_TO_RUN="${DATASETS_TO_RUN:-master external internal}"
SMOKE="${SMOKE:-0}"
LONG_HOURS="${LONG_HOURS:-68}"
SHORT_HOURS="${SHORT_HOURS:-68}"
SLOTS="${SLURM_JOB_NUM_NODES:-1}"
CPUS="${SLURM_CPUS_ON_NODE:-14}"

RESDIR="${RESDIR:-results}"
[ "$SMOKE" = "1" ] && RESDIR=results_smoke

tasks=()
for m in $MODELS_TO_RUN; do
    for v in $VARIANTS_TO_RUN; do
        for d in $DATASETS_TO_RUN; do tasks+=("$m:$v:$d"); done
    done
done
echo "[$(date +%T)] bundle job ${SLURM_JOB_ID:-?}: ${#tasks[@]} runs on $SLOTS node(s): ${SLURM_JOB_NODELIST:-?}"

run_one() {
    local m="$1" v="$2" d="$3"
    local run="$PIPE/models/$m/$RESDIR/${d}_${v}"
    local data="$DATASETS/${d}_dataset.csv"
    local hours="$SHORT_HOURS"
    case "$m" in dqn_gwo_rf|gwo_rf) hours="$LONG_HOURS" ;; esac
    mkdir -p "$run/figures" 2>/dev/null || true
    if [ -f "$run/DONE" ]; then
        echo "[$(date +%T)] $m/${d}_$v already complete"
        return 0
    fi
    if [ ! -f "$data" ]; then
        echo "missing dataset $data" > "$run/FAILED"
        mark_failed train "$m/${d}_$v"
        return 1
    fi
    rm -f "$run/FAILED"
    date +%FT%T > "$run/STARTED"
    local flags=(--data "$data" --run-dir "$run" --name "$d" --variant "$v" --max-hours "$hours")
    [ "$SMOKE" = "1" ] && flags+=(--smoke)
    echo "[$(date +%T)] start  $m/${d}_$v"
    if srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem=0 \
            "$PY" "$PIPE/models/$m/train.py" "${flags[@]}" ${EXTRA_ARGS:-} >> "$run/train.log" 2>&1; then
        echo "[$(date +%T)] done   $m/${d}_$v"
    else
        rc=$?
        echo "exit $rc at $(date +%FT%T), job ${SLURM_JOB_ID:-?}" > "$run/FAILED"
        mark_failed train "$m/${d}_$v"
        echo "[$(date +%T)] FAILED $m/${d}_$v (exit $rc)"
        return 1
    fi
}

mkdir -p "$PIPE/logs"
for t in "${tasks[@]}"; do
    IFS=: read -r m v d <<< "$t"
    while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n || true; done
    run_one "$m" "$v" "$d" &
done
wait
echo "[$(date +%T)] bundle finished"
