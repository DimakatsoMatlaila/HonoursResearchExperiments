# Honours research experiments: metadata-based detection of masquerading C2 channels

Code, profile configurations and result tables for the Wits Honours project
*Metadata-Based Detection of Masquerading Command-and-Control Channels Across Malleable Profile Configurations*.
The experiments train flow-metadata classifiers on traffic from ten Sliver C2 profiles and evaluate them with
leave-profiles-out (LPO) cross-validation: each profile is held out in turn, together with the last 20 % (in time)
of the benign traffic.

The raw packet captures and the flow datasets are **not** in this repository (see *Data*).

## Contents

| Folder | What it holds |
|---|---|
| `data_pipeline/` | Slurm scripts that filter the pcaps, label flows and build the datasets |
| `common/` | Flow-feature extraction, LPO runner, shared metrics and plotting, analysis scripts |
| `models/<model>/train.py` | One training script per model (12 models, see below) |
| `slurm/` | Job submission scripts (models, seeds, analyses) |
| `profiles/sliver/` | The ten Sliver HTTP C2 profile files used to generate the C2 traffic |
| `models/<model>/results*/` | Per-run result tables (`metrics_per_fold.csv`, `summary.txt`, `run_config.json`, search histories) |
| `analysis/` | Output of the dukes_apt29 missed-flow analysis, the cross-protocol count and the check-in checks |
| `tests/` | End-to-end test on synthetic pcaps (`bash tests/run_local_tests.sh`) |
| `config.sh`, `menu.sh`, `track.sh` | Paths and settings, an interactive menu and a job tracker |

Checkpoints, figures and training logs are excluded to keep the repository small (see `.gitignore`).

## Setup

The scripts were written for the Wits `mscluster` Slurm cluster. All heavy work is submitted with Slurm; nothing runs on
the login node. Paths are set in `config.sh` and can be overridden with environment variables:

```bash
export RAW=/path/to/pcaps_raw WORK=/path/to/work PY=/path/to/python
```

Python environment: numpy, pandas, scikit-learn, scipy, matplotlib, dpkt, torch, xgboost.
`config.sh` also lists nodes that were excluded because of hardware faults (`EXCLUDE_NODES`); adjust for your cluster.
`c2_ips.txt` maps each profile to the C2 ingress address used for labelling (a flow is malicious if either endpoint
is a C2 address).

## Data

The pipeline reads the internal (pre-NAT) and external (post-NAT) captures from `$RAW`; the expected layout is built by
`data_pipeline/make_manifest.sh`. Captures not named `internal_*` or `external_*` are ignored.

```bash
bash menu.sh                       # option 1: clean, label and build datasets
# or: bash data_pipeline/run_data_pipeline.sh
```

Output: `internal_dataset.csv`, `external_dataset.csv`, `master_dataset.csv` (both taps) with 25 numeric flow features
(`mixed` variant) and a TLS-only variant (`--variant https`, 23 features once constant columns are dropped).

## Models

Twelve models: `rf_baseline`, `hist_gb`, `xgb` (tree ensembles); `gwo_rf`, `dqn_gwo_rf` (grey-wolf feature search, plain
and with a Dueling-DQN controller); `gwo_rf_nested`, `dqn_gwo_rf_nested` (the same searches with a held-out-profile
fitness signal); `mlp_erm`, `mlp_vrex`, `mlp_gdro` (profile-invariant networks); `isolation_forest`, `lsh_rf`.

```bash
bash slurm/smoke_test.sh                                 # tiny run of every model first
bash slurm/submit_model.sh <model> [mixed|https|both]    # one model
SEED=1337 TIER=cheap bash slurm/submit_seeds.sh          # another seed; TIER = cheap | search | nested
bash slurm/submit_analysis.sh dukes                      # re-run the dukes_apt29 fold and save the missed flows
bash slurm/submit_analysis.sh cross                      # protocol counts / protocol-pair experiment
bash track.sh                                            # job and run status
```

The random seed is read from `C2_SEED` (default 42). Results go to `models/<model>/results/` for seed 42 and
`models/<model>/results_seed<S>/` for other seeds. Runs checkpoint after every fold and resume if resubmitted.
The paper uses seeds 42, 1337 and 2024.

## Results

Each finished run folder holds `metrics_per_fold.csv` (one row per held-out profile: confusion counts, precision,
recall, F1, false-positive rates with bootstrap intervals, ROC-AUC), `summary.txt`, `run_config.json` and, where
relevant, `gwo_history.csv` / `history.csv` / `feature_importance.csv`. `common/collect_results.py` and
`common/collect_seeds.py` aggregate them. The true-positive-rate-at-fixed-FPR figures in the paper are read from the
ROC curves stored in the run checkpoints, which are not part of this repository.

## Profiles

`profiles/sliver/*-c2.json` are the Sliver HTTP C2 profile files used to generate the implants. They were adapted from
published Cobalt Strike malleable profiles, and the network-visible parts (paths, headers, user agents, encodings,
certificate attributes) were carried over. The sleep and jitter settings of the originals are not verified to have been
applied by Sliver; see the paper. Implant binaries are not included.

## Ethics and use

All traffic was generated in an isolated lab with no real user data. The profile files describe request shapes to be
used for detection research; do not use them against systems you do not own or have permission to test.

## AI assistance

Parts of this code was debugged  with the help of an AI assistant (Claude, Anthropic).
I did the experimental design, lab, data collection,model code, runs and verification of results
