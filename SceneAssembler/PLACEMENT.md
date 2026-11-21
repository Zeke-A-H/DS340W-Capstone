# Learned placement: LlamaFT and TrAssembler experiments

There are now two **alternative** placement predictors. Each predicts a transform
from normalized object-local CAD coordinates directly into the scene frame. Choose
one predictor's JSONL file for assembly; do not apply both transforms sequentially.
Existing LlamaFT structure prediction, TrAssembler CAD geometry, and annotated
assembly remain available without changed commands.

This is a supervised placement extension, not a claim that original Img2CAD
learned scene assembly. The current version uses **provided object boxes and
orthographic camera calibration**. Object detection, camera estimation, and
placement from an arbitrary uncalibrated photograph are not implemented.

## Data and targets

The existing train/val/test object manifests and scene annotations are used.
Combined training contains 380/110/50 crops. `object_count` filters source scene
size as in the existing CAD workflows. No new rendering or CAD preparation is
required. Train-only statistics are used by the regression head; validation and
test placement targets are never input features.

Each target consists of translation (3), rotation columns (6D), and positive
uniform scale. Rotation is decoded into SO(3) and scale is represented internally
as log scale. The target is:

```text
normalized_to_scene = source_to_scene × inverse(source_to_normalized)
```

Reference normalization is used to construct **training/evaluation targets only**.
At placement inference and predicted assembly, no reference object scale or
annotated placement is used to rescale the predicted transform.

**Dataset limitation:** all 380 current training objects have identity rotation.
Translation and scale vary, but rotation generalization cannot be studied until
rotated scenes are added. Checkpoints record the target variation statistics.

## 1. LlamaFT placement

This uses the current 4-bit vision Llama + LoRA fine-tuning recipe. The full scene
is shown with the focused object's box marked in green; the prompt supplies the
box and camera calibration. The response is placement JSON, not CAD text.

First validate processor batches on CPU (no model weights are loaded):

```bash
data/env-linux/bin/python LlamaFT/check_batch.py --task placement
```

Submit the two-step GPU smoke test:

```bash
sbatch --account=read_crch_l1ejh6103 --time=00:30:00 \
    tools/slurm-llama.sbatch LlamaFT/finetune.py \
    --task placement --category scene --smoke_test
```

After that passes, train:

```bash
sbatch --account=read_crch_l1ejh6103 --time=05:00:00 \
    tools/slurm-llama.sbatch LlamaFT/finetune.py \
    --task placement --category scene --nepochs 10
```

The time limit is an allocation cap, not a measured runtime estimate. Inspect
loss and predictions before deciding on a training budget.

Checkpoints go under `data/ckpts/llamaft_placement/scene/all_objects/`, with a
`final/` adapter and task marker. Smoke checkpoints use `llamaft_placement_smoke`.
An existing **placement** run can be resumed using `--resume_from_checkpoint`.
Optionally initialize the new adapter from the trained CAD adapter using:

```text
--init_adapter data/ckpts/llamaft/scene/all_objects/final
```

That is weights-only initialization, not a resume; the original CAD adapter is
not modified. Do not combine it with `--resume_from_checkpoint`.

Predict placements after training:

```bash
sbatch --account=read_crch_l1ejh6103 --time=00:30:00 \
    tools/slurm-llama.sbatch LlamaFT/infer.py \
    --task placement --category scene --split test --num_tokens 256 --save_text
```

Outputs: `data/output/llamaft_placement/scene/all_objects/test/predictions.jsonl`.
Raw text and invalid JSON/errors are retained. `--skip_existing` reuses valid
predictions only for matching adapter/input signatures. Placement inference
requires a placement-trained adapter marker, avoiding accidental CAD-adapter use.

Recompute metrics from the saved predictions without loading the model:

```bash
data/env-linux/bin/python LlamaFT/eval.py
```

Add `--object_count N` to training, inference, and evaluation for a specialized
condition. Their checkpoint/output groups are `<N>_objects`. Explicit adapter
paths can evaluate one trained adapter on another scene-size subset.

## 2. TrAssembler placement

Use the current Lightning trainer with a **new placement head on the same frozen
DINOv2-B/14 architecture** used by TrAssembler. It combines full-scene features,
focused-crop features, box coordinates and calibrated camera features. Its output
is placement, not CAD commands/parameters. The geometry diffusion decoder is not
retrained or loaded into this head. Uncoordinated image-affine augmentation is
not applied because it would invalidate placement targets.

CPU head/framework smoke test (uses a tiny stand-in encoder, not DINO):

```bash
data/env-linux/bin/python TrAssembler/check_placement.py
```

Then test the real DINO path in a GPU allocation:

```bash
sbatch --account=read_crch_l1ejh6103 --time=00:30:00 \
    tools/slurm-trassembler.sbatch TrAssembler/train.py \
    task=placement dataset=scenes category=scene batch_size=4 fast_dev_run=true
```

After that passes, train, for example:

```bash
sbatch --account=read_crch_l1ejh6103 --time=02:00:00 \
    tools/slurm-trassembler.sbatch TrAssembler/train.py \
    task=placement dataset=scenes category=scene batch_size=4 max_epochs=100 \
    lr=2e-4 limit_val_batches=1.0
```

100 epochs is an example, not an established convergence requirement. Placement
checkpoints are separated automatically under
`data/ckpts/trassembler_placement/scene/scenes_None/<timestamp>/`. Geometry paths
remain unchanged. The run contains `placement_config.json` with train-only
statistics. Use `resume_from_checkpoint=/path/to/checkpoints/last.ckpt` to restore
an interrupted **placement** run. Geometry checkpoints cannot initialize this new
head; both tasks use the same pretrained frozen image encoder.

Predict with the trained run directory:

```bash
sbatch --account=read_crch_l1ejh6103 --time=00:30:00 \
    tools/slurm-trassembler.sbatch TrAssembler/eval.py --task placement \
    --model_dir data/ckpts/trassembler_placement/scene/scenes_None/YOUR_TIMESTAMP
```

Outputs: `data/output/trassembler_placement/scene/all_objects/test/predictions.jsonl`.
Inference reads no reference placement targets. Evaluation reads targets only
after predictions have been generated. Recompute saved prediction metrics with:

```bash
python TrAssembler/eval.py --task placement --metrics_only
```

For count-specific models, add `object_count=N` to training and
`--object_count N` to inference/evaluation. Existing geometry dataset preparation
is not a prerequisite for the placement head; packaged CAD bounds can supply
normalization for supervised labels when prepared metadata is unavailable.

## 3. Placement metrics

Both predictors use the same evaluator and write:

- `placement_metrics.json`: coverage, overall means and scene-size/category breakdowns.
- `placement_metrics_per_object.jsonl`: individual errors and parse/validity failures.

Metrics (lower errors are better):

| Metric | Meaning |
| --- | --- |
| `valid_rate` | Fraction of expected objects with a valid, correctly keyed transform; higher is better. |
| `mean_translation_error` | Euclidean translation error in arbitrary scene units. |
| `mean_rotation_error_degrees` | Geodesic angular error between rotations, in degrees. |
| `mean_scale_relative_error` | Absolute relative scale error, e.g. 0.1 means 10% scale error. |
| `mean_log_scale_error` | Absolute log ratio of predicted/reference scale. |

Errors average valid predictions only, so always report coverage with them.
Invalid/duplicate transforms and missing predictions cannot silently become
perfect scores. These are **placement** metrics; CAD geometry quality remains
measured by the existing TrAssembler metrics and full-scene metrics.

## 4. Assemble using predicted placement

Use either model's placement JSONL, along with existing reconstructed geometry:

```bash
sbatch --account=read_crch_l1ejh6103 SceneAssembler/assemble.sbatch \
    --placement_source predicted \
    --placement_predictions data/output/trassembler_placement/scene/all_objects/test/predictions.jsonl \
    --output_dir data/output/scene_assemblies/predicted_trassembler/test
```

For LlamaFT placement, replace the JSONL path with
`data/output/llamaft_placement/scene/all_objects/test/predictions.jsonl` and use a
separate `predicted_llama/test` output root for comparison. These separate folders
prevent replacing the annotated-placement baseline. The input geometry can be
selected independently via `--prediction_mode specialized` or `--predictions_dir`.

`--placement_source annotated` remains the default. Predicted mode applies the
predicted `normalized_to_scene` matrix **directly**, does not invert reference
normalization again, and never substitutes an annotated pose if a prediction is
missing. Scene/object grouping and crops still come from the dataset annotations.
A full-scene metric can now reflect placement errors as well as geometry errors;
compare it with the annotated-placement baseline while holding geometry fixed.

## 5. CPU integration checks

```bash
data/env-linux/bin/python SceneAssembler/check_placement.py
```

Checks all 540 target transforms, round trips/JSON parsing, inference-target
separation, train-only statistics, invalid/missing/duplicate prediction coverage,
and verifies that a deliberately changed predicted placement reaches the
assembler without ground-truth fallback. Perfect-target tests are synthetic
validation checks, not performance results of a trained model.

No placement model is already trained just because these checks pass. Run each
real GPU smoke test before spending credits on full training.
