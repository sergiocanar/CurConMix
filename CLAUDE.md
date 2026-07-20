# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CurConMix (MICCAI 2025): a curriculum contrastive-learning + feature-mixup framework for surgical
action triplet `<instrument, verb, target>` recognition on the CholecT45 dataset. Pure research
training code (no package, no tests, no CI) driven by Hydra config composition on the CLI.

## Setup

```bash
pip install -r requirements.txt
```

Python env used originally is Windows-flavored (pywin32 in requirements.txt, hardcoded
`C:/Users/...` and `E:/...` paths in a couple of scripts — see Known rough edges below). No
build step; scripts are run directly with `python`.

## Data layout expected on disk

All paths are resolved relative to `CFG.parent_path` (default `data`), configured in `config.yaml`:

- `data/<train_path>/VIDxx/*.png` — CholecT45 frames per video (get from
  https://github.com/CAMMA-public/cholect45)
- `data/triplet/`, `data/instrument/`, `data/verb/`, `data/target/` — per-video CSV annotation
  files consumed by `parse.py`
- `data/dataframes/CholecT45.csv` — the single flattened dataframe (metadata + one-hot triplet/i/v/t
  columns) that every other script reads; produced once by `parse.py` and cached
- `data/phase_annotations/` — optional Cholec80 phase labels, only used if `CFG.phase=true`

Run `python parse.py` once to generate `data/dataframes/CholecT45.csv` before anything else.

## Pipeline & commands

Everything is invoked as `python <script>.py key=value ...` (Hydra CLI overrides against
`config.yaml`, which lives at repo root — no `conf/` subfolder). `main.py` is the single entry
point for both pre-training and fine-tuning; which path runs depends on `ssl=True/False`.

The full pipeline (see `run_curconmix.sh` / `RUN.sh` for canonical invocations):

```bash
# 1. Pre-training: curriculum contrastive learning with feature mixup
python main.py epochs=3 target_size=131 ssl=True exp=CurConMix_Pretraining \
    split_selector='cholect45-crossval' device='cuda:0' \
    method='curriculum_supcon' ssl_loss='supcon' feature_batch=True feature_mixup=True \
    alpha=0.4 label_order='[t,it,ivt]'

# 2. Fine-tune teacher with input mixup (loads pretraining checkpoint via pretrained_ssl=True)
python main.py epochs=20 target_size=131 exp=Teacher_IM04 pretrained_ssl=True \
    pretrained_exp=CurConMix_Pretraining_2 method='supcon' \
    split_selector='cholect45-crossval' device='cuda:0' mixup=True alpha=0.4

# 3. Generate teacher soft-labels for self-distillation
python generate.py target_size=131 exp=Teacher_IM04 split_selector='cholect45-crossval' \
    inference=False device='cuda:0'

# 4. Train student with self-distillation + mixup
python main.py epochs=40 target_size=131 exp=Student_IM04 distill=True \
    split_selector='cholect45-crossval' teacher_exp=Teacher_IM04 device='cuda:0' \
    mixup=True alpha=0.4

# 5. Run inference (predictions, not soft-labels) for both models
python generate.py target_size=131 exp=Teacher_IM04 split_selector='cholect45-crossval' \
    inference=True device='cuda:0'
python generate.py target_size=131 exp=Student_IM04 split_selector='cholect45-crossval' \
    inference=True device='cuda:0'

# 6. Evaluate: computes official CholecT45 mAP over saved prediction CSVs
python evaluate.py inference=true
```

Notes on the pipeline:
- `exp=<name>` pretraining checkpoints are saved with an implicit `_<epoch>` suffix (see
  `train_cross_val_SSL` in `train.py`) — that's why fine-tuning references
  `pretrained_exp=CurConMix_Pretraining_2` (epoch index 2) rather than the bare exp name used at
  pretraining time.
- `target_size=131` is the multitask head size (100 triplet classes + component classes);
  `evaluate.py` internally forces `target_size=100` since only triplet columns are scored.
- There's no single training-vs-fine-tuning flag; the flow through `main.py` branches on `CFG.ssl`,
  and within fine-tuning, further branches on `CFG.mixup` / `CFG.distill` (see Architecture below).
- `debug=true` forces `epochs=1` and disables Neptune logging, for a fast smoke pass.
- No automated test suite exists. "Testing" a change means running a short `epochs=1 debug=true`
  pass through the relevant stage and checking the printed loss / mAP.

## Architecture

### Config-driven dispatch, not OOP

There's no trainer class hierarchy — one Hydra `CFG` object threads through every function, and
behavior is selected by large if/elif chains keyed on `CFG.ssl`, `CFG.method`, `CFG.feature_batch`,
`CFG.mixup`, `CFG.distill`. When adding a new training mode, follow this pattern rather than
introducing a class: add a new branch in `train_cross_val_SSL`/`train_cross_val` (`train.py`), plus
a new `train_*` epoch-loop function in `helper.py` if the batch shape differs.

### Two orthogonal training regimes (`train.py`)

- `train_cross_val_SSL(CFG)` — self-supervised/contrastive **pre-training** stage (`CFG.ssl=True`).
  Builds one of `supcon_Model` / `FeatureSupConModel` (`models.py`), runs curriculum sampling by
  swapping the dataset's label granularity (`it`/`t`/`ivt`, controlled by `CFG.label_order`) at
  epoch boundaries via `train_dataset.set_label_key(...)`.
- `train_cross_val(CFG)` — supervised **fine-tuning** stage (`CFG.ssl=False`, the default). Builds
  `TripletModel`, optionally seeds it from an SSL checkpoint (`CFG.pretrained_ssl`), optionally
  blends in teacher soft-labels (`CFG.distill` → `apply_self_distillation` in `helper.py`), and
  trains with either `train_fn` or `train_mixup` (`helper.py`) depending on `CFG.mixup`.

Both loops iterate `CFG.n_fold` folds (5-fold CV over `CFG.trn_fold`), each fold reloading a fresh
model, saving `output_dir/checkpoints/fold{fold}_..._{exp}.pth` on best validation mAP
(fine-tuning) or at the final epoch (pretraining).

### Curriculum contrastive learning + feature mixup (the paper's core contribution)

`SupConFeatureMixupBatchDataset` (`dataset.py`) is the dataset used when
`feature_batch=True and feature_mixup=True`. For each anchor image it:
1. Looks up precomputed backbone features + a cosine-similarity matrix (pickled, produced offline
   by `generate_fold_features_torchvision.py`) for the *current* label granularity
   (`set_label_key`, switched over epochs to realize the curriculum: coarse → fine, per
   `CFG.label_order`).
2. Selects the single **hardest positive** (same class, lowest similarity) and top-N hardest
   **negative candidates** (different class, highest similarity), then synthesizes harder negatives
   by beta-mixing pairs of negative features (`lam ~ Beta(alpha, alpha)`).
3. Returns `(image, label, contrast_features, contrast_labels)` consumed by
   `train_feature_batch_supcon` (`helper.py`) and `FeatureBatchdSupConLoss` (`losses.py`) — a SupCon
   variant that contrasts the live image embedding against these precomputed/mixed feature vectors
   instead of another augmented view in the same batch.

`Supcon_TrainDataset` is the simpler two-view SupCon dataset (no precomputed features) used when
`feature_batch=False`, paired with `train_supcon` + `SupConLoss`.

`generate_fold_features_torchvision.py` is a standalone offline script (not wired into
`main.py`/Hydra dispatch — has its own `argparse` + YAML loader) that must be run per fold before
curriculum pretraining to produce the `*_trainset_features_*.pkl` and `*_similarity_matrix.pkl`
files that `SupConFeatureMixupBatchDataset` expects. **The output paths in
`train_cross_val_SSL` (`train.py`) and in this script are hardcoded Windows drive paths
(`C:/Users/...`, `E:/...`) — update these to your environment before running pretraining with
`feature_batch=True`.**

### Self-distillation + mixup fine-tuning

`apply_self_distillation` (`helper.py`) blends a teacher's soft-label predictions
(`softlabels/sl_f{fold}_..._{teacher_exp}.csv`, produced by `generate.py` with `inference=False`)
into the student's hard labels: `label = (1-SD)*hard + SD*soft`, optionally followed by label
smoothing (`CFG.smooth`/`CFG.ls`). `train_mixup` additionally mixes input images/labels
(`mixup_data`/`mixup_criterion`) on top of this. Both are orthogonal knobs on the same
`TripletModel` fine-tuning loop.

### Data flow / label taxonomy

`preprocess.get_folds` reads the flattened `CholecT45.csv`, derives categorical class labels for
every component combination (`i`, `t`, `v`, `it`, `iv`, `tv`, `ivt`) from the one-hot columns, and
assigns each video to one of 5 folds via the fixed `split_selector` video-ID lists (currently only
`'cholect45-crossval'` is defined — extend `split_selector()` in `preprocess.py` to add e.g. a
challenge split). Column layout in the dataframe (used throughout via `df.columns.get_loc(...)`
rather than named access): `tri0..tri99` (triplet one-hot), `inst0..inst5`, `v0..v9`, `t0..t14`,
then derived categorical columns.

### Evaluation

`utils.py` wraps `ivtmetrics.recognition.Recognition` (vendored under `ivtmetrics/`, the official
CholecT45 metric implementation) to compute **video-aggregated** mAP (`cholect45_ivtmetrics_mAP`,
used for per-epoch validation and final CV) vs. per-component breakdowns
(`cholect45_ivtmetrics_mAP_all`, used by `evaluate.py --evaluate_all` for classwise AP CSVs per
component `i/v/t/iv/it/ivt`). Video aggregation matters: metrics are computed by feeding
per-frame predictions into `rec.update()` grouped by video and calling `rec.video_end()`, not by
naively averaging per-frame AP.

### Output directory structure (`CFG.output_dir`, default `outputs/`)

- `checkpoints/fold{N}_{model_prefix}_{exp}[_{epoch}].pth`
- `softlabels/sl_f{fold}_{model_prefix}_{target_size}_{exp}.csv` — teacher predictions on train set
- `predictions/{model_prefix}_{target_size}_{exp}.csv` — predictions on held-out fold(s), consumed
  by `evaluate.py`
- `summary_dir/`, `outputs/<exp>/<exp>_fold_{N}.csv` — per-epoch loss/mAP logs

## Known rough edges (be aware when touching these areas)

- `generate_fold_features_torchvision.py` and the `feature_batch=True` branch of
  `train_cross_val_SSL` hardcode Windows paths (`C:/Users/kyuhw/...`, `E:/Surgical/...`,
  `E:/cholect45_features`) — these need to be pointed at real paths, not left as-is.
- `TrainDataset.__getitem__` checks `if self.feature != None` on a `None`-initialized attribute
  that is never populated elsewhere — effectively dead code.
- Neptune logging (`CFG.neplog`) uses the legacy `neptune.new` API; `neptune-client==0.14.2` is
  pinned in requirements.txt for this reason.
