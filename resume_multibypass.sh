#!/usr/bin/env bash
# Resume run_multibypass.sh from step 3 onward (soft-labels -> student ->
# inference -> evaluate). Use this when pretraining + teacher fine-tuning
# already completed (checkpoints exist under $OUTPUT_DIR/checkpoints) but a
# later step crashed, e.g. the PyTorch 2.6 torch.load(weights_only=True)
# default that broke soft-label generation on 2026-07-09 (fixed in
# generate.py / train.py, but re-run needed to pick up the fix).
set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cuda:0}
TAG=${TAG:-20260709}

TEACHER_EXP=${TAG}_MultiBypass_Teacher_IM04
STUDENT_EXP=${TAG}_MultiBypass_Student_IM04

# 3. Generate teacher soft-labels for self-distillation
python generate.py --config-name=config_multibypass exp="$TEACHER_EXP" \
    inference=False device="$DEVICE"

# 4. Student self-distillation + mixup
python main.py --config-name=config_multibypass epochs=40 exp="$STUDENT_EXP" \
    distill=True teacher_exp="$TEACHER_EXP" mixup=True alpha=0.4 device="$DEVICE"

# 5. Inference (predictions) for both models
python generate.py --config-name=config_multibypass exp="$TEACHER_EXP" \
    inference=True device="$DEVICE"
python generate.py --config-name=config_multibypass exp="$STUDENT_EXP" \
    inference=True device="$DEVICE"

# 6. Evaluate: official (video-aggregated) triplet mAP over both folds
python evaluate.py --config-name=config_multibypass inference=true

echo "Done. Experiments: teacher=$TEACHER_EXP student=$STUDENT_EXP"
