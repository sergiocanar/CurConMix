#!/usr/bin/env bash
# Full CurConMix pipeline on MultiBypassT40: curriculum contrastive pretraining
# -> teacher fine-tune (input mixup) -> self-distillation -> student -> eval.
#
# Requires prepare_multibypass_data.sh to have been run once already
# (MultiBypassT40.csv + features/*.pkl must exist).
set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cuda:0}
TAG=${TAG:-$(date +%Y%m%d)}

PRETRAIN_EXP=${TAG}_MultiBypass_Pretraining
TEACHER_EXP=${TAG}_MultiBypass_Teacher_IM04
STUDENT_EXP=${TAG}_MultiBypass_Student_IM04

# 1. Curriculum contrastive pretraining (3-epoch curriculum: t -> it -> ivt)
python main.py --config-name=config_multibypass epochs=3 ssl=True \
    exp="$PRETRAIN_EXP" method='curriculum_supcon' ssl_loss='supcon' \
    feature_batch=True feature_mixup=True alpha=0.4 label_order='[t,it,ivt]' \
    device="$DEVICE"

# 2. Teacher fine-tune with input mixup.
#    train_cross_val_SSL appends "_<final_epoch>" to the checkpoint name
#    (epochs=3 -> final epoch index 2), so pretrained_exp needs that suffix.
python main.py --config-name=config_multibypass epochs=20 exp="$TEACHER_EXP" \
    pretrained_ssl=True pretrained_exp="${PRETRAIN_EXP}_2" method='supcon' \
    mixup=True alpha=0.4 device="$DEVICE"

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

echo "Done. Experiments: pretrain=$PRETRAIN_EXP teacher=$TEACHER_EXP student=$STUDENT_EXP"
