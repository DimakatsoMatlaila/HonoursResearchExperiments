#!/bin/bash
cd "$(dirname "$0")"
source ./config.sh

show_menu() {
    cat <<M

${BLU}C2 metadata-detection pipeline${RST}   ${DIM}(node $EXCLUDE_NODES excluded)${RST}

 ${YEL}Data${RST}
   1) Run everything: clean, label, build datasets
   2) Clean only          3) Label only          4) Build datasets only
   5) Find a profile's C2 IP
 ${YEL}Training${RST}
   6) Smoke test all models (tiny, 1 h)
   7) Train one model     8) Train all models
   b) Train all models in ONE 6-node job
 ${YEL}Status${RST}
   9) Track progress      w) Track progress, live
  10) Collect results and figures
  11) Check python environment
   q) Quit
M
}

while true; do
    show_menu
    read -rp "> " c
    case "$c" in
        1) bash data_pipeline/run_data_pipeline.sh ;;
        2) STAGES="clean" bash data_pipeline/run_data_pipeline.sh ;;
        3) STAGES="label" bash data_pipeline/run_data_pipeline.sh ;;
        4) STAGES="build" bash data_pipeline/run_data_pipeline.sh ;;
        5) read -rp "profile: " p
           srun -p bigbatch -c 1 --exclude="$EXCLUDE_NODES" bash data_pipeline/diagnose.sh "$p" ;;
        6) bash slurm/smoke_test.sh ;;
        7) echo "models: $MODELS"
           read -rp "model: " m
           read -rp "variant [mixed|https|both]: " v
           read -rp "datasets (enter = all): " d
           bash slurm/submit_model.sh "$m" "${v:-both}" "${d:-$DATASET_NAMES}" ;;
        8) bash slurm/submit_all_models.sh ;;
        b) bash slurm/submit_bundle.sh ;;
        n) bash slurm/submit_new.sh ;;
        9) bash track.sh ;;
        w) bash track.sh watch ;;
        10) mkdir -p logs/train
            id=$(cd logs/train && sbatch --parsable --export=ALL "$PIPE/slurm/collect.sbatch")
            log_job collect "results summary" "$id"
            echo "collect job $id" ;;
        11) srun -p bigbatch -c 1 --exclude="$EXCLUDE_NODES" "$PY" common/env_check.py ;;
        q) exit 0 ;;
    esac
done
