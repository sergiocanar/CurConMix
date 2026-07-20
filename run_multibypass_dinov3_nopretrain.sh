#!/usr/bin/env bash
# Ablation: same DINOv3-base teacher/student recipe as run_multibypass_dinov3.sh,
# but WITHOUT the CurConMix curriculum contrastive pretraining stage (steps 0-1
# skipped, pretrained_ssl left at its default False so TripletModel starts from
# plain timm/DINOv3 pretrained weights). Isolates whether curriculum SupCon
# pretraining actually helps on top of DINOv3's own self-supervised features.
#
# Writes to the same output_dir/predictions as run_multibypass_dinov3.sh, so the
# final evaluate.py call reports both ablation and non-ablation experiments
# side by side.
set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cuda:7}
TAG=${TAG:-$(date +%Y%m%d)}

TEACHER_EXP=${TAG}_MultiBypass_DINOv3_NoPretrain_Teacher_IM04
STUDENT_EXP=${TAG}_MultiBypass_DINOv3_NoPretrain_Student_IM04

# 1. Teacher fine-tune with input mixup, directly from DINOv3's own pretrained
#    weights (pretrained_ssl=False, the default) -- no curriculum SupCon checkpoint.
python main.py --config-name=config_multibypass_dinov3 epochs=20 exp="$TEACHER_EXP" \
    method='supcon' mixup=True alpha=0.4 device="$DEVICE"

# 2. Generate teacher soft-labels for self-distillation
python generate.py --config-name=config_multibypass_dinov3 exp="$TEACHER_EXP" \
    inference=False device="$DEVICE"

# 3. Student self-distillation + mixup
python main.py --config-name=config_multibypass_dinov3 epochs=40 exp="$STUDENT_EXP" \
    distill=True teacher_exp="$TEACHER_EXP" mixup=True alpha=0.4 device="$DEVICE"

# 4. Inference (predictions) for both models, both folds
python generate.py --config-name=config_multibypass_dinov3 exp="$TEACHER_EXP" \
    inference=True device="$DEVICE"
python generate.py --config-name=config_multibypass_dinov3 exp="$STUDENT_EXP" \
    inference=True device="$DEVICE"

# 5. Evaluate: prints mAP for every experiment in outputs_multibypass_dinov3/predictions,
#    i.e. this ablation's teacher/student alongside the pretrained-then-fine-tuned ones.
python evaluate.py --config-name=config_multibypass_dinov3 inference=true

echo "Done. Experiments: teacher=$TEACHER_EXP student=$STUDENT_EXP"
