# TrAssembler for scene-derived objects

Stage 2 reconstructs each object's continuous CAD geometry from its crop and
LlamaFT structure prediction. Crops and categories currently come from ground
truth boxes. Scene placement is not predicted or applied yet. Every output keeps
its sample ID, scene ID, category, and annotation path for later scene assembly.

Run from the project root in the Linux environment. Keep LlamaFT training in its
existing allocation; do not start a second GPU workload in that allocation.

## Prepare continuous targets

```bash
python TrAssembler/dataset.py --dataset scenes
```

This reads the packaged scene CAD targets and crops directly, computes normalized
continuous CAD targets and CLIP part embeddings, and saves samples under
`data/trassembler_data/scene/all_objects/`. Images are resized to 400x400 and
normalized to [-1, 1]. The original part and command ordering is preserved.
Preparation reuses existing frozen CLIP text embeddings when available; missing
embeddings are computed on CPU. CLIP weights may download on first use into
`data/.cache/clip/`. No original images or continuous targets are reused.
`train.json`, `val.json`, and `test.json` retain the scene records. Normalization
statistics are computed **only from training targets**, using training masks and
angle conversion. Existing source PKLs are not required.

To prepare a count-specific experiment, add `--object_count 3`. Its data go under
`data/trassembler_data/scene/3_objects/`. Preparation may overwrite matching
prepared files; it does not modify the packaged scene targets or LlamaFT outputs.

## Train

First check one train and validation batch inside a separate GPU allocation:

```bash
bash tools/run-linux.sh TrAssembler/train.py dataset=scenes category=scene batch_size=1 fast_dev_run=true
```

Then train a shared model across all scene sizes:

```bash
bash tools/run-linux.sh TrAssembler/train.py dataset=scenes category=scene batch_size=1 max_epochs=100
```

Or submit training as its own batch job (five-hour V100 allocation):

```bash
mkdir -p logs
sbatch tools/slurm-trassembler.sbatch TrAssembler/train.py dataset=scenes category=scene batch_size=1 max_epochs=100
```

The five-hour limit is an allocation cap, not a runtime estimate for 100 epochs.

For a count-specific model, prepare that count first and add `object_count=3` to
the training command. The default `object_count=null` pools all counts. Training
uses the train split, and validation uses val, not test. Scene validation reports
loss without the original dataset's visualization, which assumes numeric IDs.
100 epochs is an example, not a measured convergence requirement.

Each timestamped run contains `.hydra/config.yaml`, `checkpoints/last.ckpt`,
logs, and a copy of `mean_scale_stats.json`. The experiment path is printed at
startup, under `data/ckpts/trassembler/scene/scenes_None/` for combined training
(or `scenes_3/` for count 3).

Training starts from scratch unless `+ckpt=/path/to/checkpoint.ckpt` is supplied.
When fine-tuning existing weights, explicitly set
`mean_scale_stats_path=/path/to/the/checkpoints/original/statistics.json` so their
normalization convention is retained. Loading weights does not resume optimizer
state. Original category-specific training remains available with default config.

## Resume an interrupted training run

Use `resume_from_checkpoint=/path/to/checkpoints/last.ckpt` for full restoration
of model weights, optimizer state, and epoch/step progress. Keep the original
dataset, batch size, architecture, and normalization statistics. Set
`mean_scale_stats_path` to the original run's `mean_scale_stats.json`, and
`hydra.run.dir` to that same run directory to continue saving there.
`max_epochs=150` means 150 total epochs, including completed epochs. Do not combine
`resume_from_checkpoint` with the weights-only `+ckpt` option.

## Infer using LlamaFT predictions

After both stages have trained, use the timestamped TrAssembler run directory:

```bash
bash tools/run-linux.sh TrAssembler/eval.py --dataset scenes \
    --model_dir data/ckpts/trassembler/scene/scenes_None/YOUR_RUN_TIMESTAMP \
    --max_samples 1
```

After inspecting that result, omit `--max_samples 1` to process the split:

```bash
bash tools/run-linux.sh TrAssembler/eval.py --dataset scenes \
    --model_dir data/ckpts/trassembler/scene/scenes_None/YOUR_RUN_TIMESTAMP
```

Default inputs: `data/output/llamaft_h5/scene/all_objects/test/`.
Default outputs: `data/output/trassembler_h5/scene/all_objects/test/<sample_id>/`:

- `final.h5`: continuous CAD vectors, including individual part datasets.
- `final.step`: editable CAD exchange output, if solid conversion succeeds.
- `final.obj`: mesh output.
- `final.png`: crop/reconstruction preview (unless rendering is disabled or fails).
- `metadata.json`: scene association, checkpoint, status, and any error.

`predictions.jsonl` records successes and failures after each processed object.
A nonzero exit code indicates at least one failed object; its metadata explains
why. An HDF5 file can exist even if conversion to a solid failed, so use the
recorded status when assessing reconstruction success.

Add `--object_count 3` to evaluate only crops from three-object scenes. By default
this also selects that count's LlamaFT output folder. To evaluate the combined
LlamaFT adapter on those crops, explicitly use
`--predictions_dir data/output/llamaft_h5/scene/all_objects/test/`.
Use `--output_dir` to separate comparisons between checkpoints. Only use
`--skip_processed` when keeping the same checkpoint and inference settings.

## Original evaluation features on scene crops

Scene inference now uses the same geometry metric functions and category-specific
PointNet2 segmentation evaluator as the original path. It calculates geometry
metrics after reconstruction automatically. You can also evaluate existing files
without rerunning reconstruction or loading TrAssembler:

```bash
python TrAssembler/eval.py --dataset scenes --metrics_only
```

For all original metric families (OCC, CD, part accuracy, and part mIoU), use a
GPU allocation for the PointNet2 evaluation. From the login node:

```bash
sbatch --account=read_crch_l1ejh6103 --time=00:30:00 \
    tools/slurm-trassembler.sbatch \
    TrAssembler/eval.py --dataset scenes --metrics_only --eval_partseg
```

For each specialized experiment's existing outputs:

```bash
for count in 1 2 3 4; do
    sbatch --account=read_crch_l1ejh6103 --time=00:30:00 \
        tools/slurm-trassembler.sbatch \
        TrAssembler/eval.py --dataset scenes --object_count "$count" \
        --metrics_only --eval_partseg
done
```

Metrics are saved in the existing prediction folder, e.g.
`data/output/trassembler_h5/scene/all_objects/test/` or
`data/output/trassembler_h5/scene/3_objects/test/`:

- `metrics.json`: overall results, breakdowns by scene object count and furniture
  category, segmentation coverage and per-category results, and metric definitions.
- `metrics_per_object.csv`: individual object scores and errors for spreadsheets.
- `metrics_per_object.jsonl`: the same per-object scores as JSON records.

Metric interpretation:

| Metric | Interpretation |
| --- | --- |
| `mean_occ` | Successful reconstructed mesh fraction, 0–1; higher is better. This is the original code's generation-success metric, not volumetric overlap. |
| `mean_cd` | Original bidirectional mean Euclidean Chamfer distance on 2048 surface points after independently centering and unit-radius-normalizing each shape; lower is better. It is not a percentage, squared distance, or scene-placement score. |
| `partseg_accuracy` | Original PointNet2 part-label agreement, 0–1; higher is better. |
| `partseg_instance_avg_iou` | Original instance-averaged part IoU, 0–1; higher is better. |

CD excludes failed/missing measurements; always report `mean_occ` and
`valid_cd_count` alongside it. Segmentation uses original reference IDs for
annotations and category checkpoints, while keeping scene instance IDs distinct.
The original chair/table coordinate transform and storage part-label mask are
retained. Missing annotations or checkpoints are reported as unavailable, not
as zero accuracy. Null segmentation scores mean not requested or unavailable;
check `partseg_requested`, coverage counts, and `part_segmentation` errors.
Recomputing without `--eval_partseg` produces geometry-only reports.

To match the original README inference settings, `--original_eval` enables
text retrieval, reference symmetry refinement, and reference arc assistance
together. First audit prerequisites on the login node:

```bash
python TrAssembler/eval.py --dataset scenes --original_eval --check_inputs
```

The audit saves `evaluation_inputs.json` in the existing prediction folder.
A nonzero exit means input prerequisites are missing. It does not load model
weights or reconstruct anything. The current test set has reference symmetry
labels for only 15/50 crops; the other 35 must be supplied before a full
original-style comparison. Missing labels are never replaced with "no symmetry".
Add `--object_count N` to audit a specialized condition. This is a data-label
requirement, not a reason to retrain either stage.

After prerequisites are complete, use `--original_eval` on the normal inference
command with the matching `--model_dir`. This regenerates predictions; adding it
to `--metrics_only` cannot retrospectively refine existing reconstructions. In
metrics-only mode, the preset rejects outputs whose saved options do not match.
Use the same protocol for all training conditions when comparing them.

The following original inference options now work for scenes:

| Option | Scene behavior |
| --- | --- |
| `--text_emb_retrieval` | Uses the furniture category's original part-name dictionary and shared sample builder. |
| `--save_intermediate` | Saves per-step CAD vectors, meshes, and previews using the original callback. |
| `--skip_processed` | Reuses completed outputs only when checkpoint timestamp and inference options also match. |
| Final preview | Saves `final.png` with input crop and rendered reconstruction; `--no_preview` disables rendering. Preview failures are recorded separately from CAD conversion failures. |
| `--eval_partseg` | Routes each crop through its category's original PointNet2 checkpoint and reference annotations. |
| `--symmetry_refinement` | Enables the original symmetry refinement using `--sym_labels_dir/<reference_id>.txt`. This is reference-assisted, so it is opt-in. |
| `--use_gt_arc_args` | Also supplies original-style reference arc direction during refinement. Requires `--symmetry_refinement`; explicitly a ground-truth-assisted experiment. |

Without the last two options, prediction does not read ground-truth CAD or symmetry
labels; reference meshes/annotations are only read for evaluation afterward.
The furniture category itself still comes from the dataset, not a predicted classifier.
`--category chair`, `table`, or `storagefurniture` can restrict scene evaluation;
`--category scene` is an alias for the mixed-category scene workflow.

The storage segmentation checkpoint provided in this repository has a 50-label
head, despite the original loader constructing a 7-label model. The loader now
uses the checkpoint's actual head/conditioning dimensions and applies the original
seven storage labels and three-part IoU mask; it does not discard checkpoint weights.


## CPU validation

```bash
python TrAssembler/check_scene.py
```

Checks prepared samples, part/command correspondence, image shapes, variable-size
batching, vector decoding, count filtering, training-only statistics, metric
coverage/failure accounting, deterministic sampling, scene-instance/reference
mapping, and original/scene command routing. GPU rendering/refinement still
requires a GPU runtime test.

## Scene assembly boundary

The reconstructed geometry is in **normalized object-local coordinates**. The
prepared metadata records the source-to-normalized scale; it must be inverted
before applying the dataset's source-to-scene transform in a future assembly
stage. Applying scene transforms directly to normalized objects would give the
wrong sizes. Full-scene CAD export using annotated placement is implemented in
[SceneAssembler](../SceneAssembler/README.md). Automatic placement estimation is
not implemented. TrAssembler remains the per-object reconstruction stage.

Object count is a dataset filter, not a direct whole-scene model input. Scene
crops may include occlusions and changed viewpoints; geometry quality needs GPU
experiments and held-out evaluation. A successful file conversion is not an
accuracy score.

## Learned placement task

See [placement training and evaluation](../SceneAssembler/PLACEMENT.md) for the new
placement task in both model families. Existing CAD training/inference commands
remain the default. Placement checkpoints are separate and require new training.
