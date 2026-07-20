#!/usr/bin/env bash
# One-time setup for MultiBypassT40: convert JSON annotations to the CSV the
# pipeline expects, then extract backbone features + cosine-similarity matrices
# needed by curriculum pretraining (feature_batch=True, feature_mixup=True).
#
# Already run once for the full dataset — MultiBypassT40.csv and features/*.pkl
# exist. Re-run only if the dataset changes (re-parse is fast; feature
# extraction takes ~15-20 min per fold and writes a ~7GB similarity matrix
# per fold, so it's split out from run_multibypass.sh on purpose).
set -euo pipefail
cd "$(dirname "$0")"

python parse_multibypass.py

python generate_fold_features_torchvision.py --cfg config_multibypass.yaml
