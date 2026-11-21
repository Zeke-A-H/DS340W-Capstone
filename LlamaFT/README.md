# LlamaFT scene experiments

Use the GPU allocation and Linux environment described in the project README.
Keep the same `HF_HOME` used to download the base Llama model. No command below
is started automatically by the data preparation or batch checks.

## Stage 1 selection by source-scene object count

Use the same main-README workflow with `--category scene --object_count N`:

```bash
python LlamaFT/finetune.py --category scene --object_count 3 --nepochs 10
python LlamaFT/infer.py --category scene --object_count 3 --split test --save_text
```


To pool crops from **all scene sizes (1–4 objects)** into one adapter, omit
`--object_count` in both commands:

```bash
python LlamaFT/finetune.py --category scene --nepochs 10
python LlamaFT/infer.py --category scene --split test --save_text
```

The combined adapter is saved under `data/ckpts/llamaft/scene/all_objects/`
(with the final adapter in `final/`). Predictions go to
`data/output/llamaft_h5/scene/all_objects/test/`. These paths are separate from
count-specific experiments. Inference automatically selects the combined
adapter; it must be trained first. This still processes individual object crops,
not whole-scene images. Training uses the train/validation splits, while the
inference command above uses only the test split.

Counts 1–4 are supported. This pools furniture types and selects individual
object crops from scenes of that size; whole-scene prediction is not enabled.
For N=3, train/validation/test contain 114/33/15 crops (38/11/5 source scenes).
Checkpoints: `data/ckpts/llamaft/scene/3_objects/`, with final adapter in `final/`.
Predictions: `data/output/llamaft_h5/scene/3_objects/test/`.
Add `--smoke_test` to train two steps in a separate `llamaft_smoke` checkpoint
root. Explicit `--output_dir` still sets the training checkpoint root.
The single shared prompt is unchanged. Command order is preserved for these
adapters. Stage 2 scene-crop training and inference are documented in [TrAssembler](../TrAssembler/README.md).

The earlier `--dataset scenes --category chair` commands below remain available
for category-specific crop experiments. Their checkpoint paths are different.

## First: two-step GPU training test

```bash
bash tools/run-linux.sh LlamaFT/finetune.py --dataset scenes --category chair --smoke_test
```

This starts a new adapter from the base vision-language model, uses four training
and four validation crops, runs two optimizer steps, evaluates/saves each step,
and checks finite losses and a changed LoRA matrix. It uses batch size 1 and
native BF16 when supported, otherwise FP16 (including V100). Override using
`--precision fp16` or `--precision bf16`; unsupported native BF16 is rejected.
Training stays on the current GPU rather than automatically offloading layers.
Actual GPU memory usage and training stability must be established by this test.

Smoke output: `data/ckpts/llamaft_scenes_smoke/chair/`, with a `final/` adapter
and processor. This is separate from full experiment checkpoints. Full training
retains TensorBoard logging; the smoke test disables external logging.

## Full scene-crop training, after the test succeeds

```bash
bash tools/run-linux.sh LlamaFT/finetune.py --dataset scenes --category chair --nepochs 10
```

Outputs are in `data/ckpts/llamaft_scenes/chair/`. The `final/` subdirectory
contains the final adapter and processor. Checkpoints are also saved every 25 optimizer steps
(at most two retained); final does not mean the best validation checkpoint.
`--batch_size` defaults to 1. `--dataset original` retains original-image loading,
which requires the original text-label files to be present. Train/validation
examples use the same main prompt, and loss supervises only the CAD response.

## Evaluate transfer in either direction

Original pretrained adapter on held-out scene crops (no scene training required):

```bash
bash tools/run-linux.sh LlamaFT/infer.py --dataset scenes --adapter_source original --category chair --split test --max_samples 1 --save_text
```

Scene-trained adapter on held-out original images:

```bash
bash tools/run-linux.sh LlamaFT/infer.py --dataset original --adapter_source scenes --category chair --split test --max_samples 1 --save_text
```

Scene-trained adapter on scene crops:

```bash
bash tools/run-linux.sh LlamaFT/infer.py --dataset scenes --category chair --split test --max_samples 1 --save_text
```

Adapter selection prefers `final/`, otherwise the highest numeric checkpoint.
Use `--adapter_path` to choose a specific checkpoint, such as the smoke adapter.
Set `--adapter_source scenes` when explicitly selecting a scene-trained adapter:
this automatically disables legacy arc reordering. Original adapters retain
legacy parsing unless `--preserve_command_order` is supplied explicitly.

Cross-dataset outputs are separated under
`data/output/llamaft_<adapter_source>_on_<dataset>_h5/<category>/<split>/`.
Original-on-original inference keeps the original `llamaft_h5/<category>/` path.
These HDF5 files are structure predictions, not final geometric reconstructions
or 3D accuracy scores. TrAssembler crop integration is available through `TrAssembler/eval.py --dataset scenes`;
scene assembly remains a separate stage. Ground-truth boxes/categories are supplied for scene crops. Published
checkpoint training membership has not been checked against the new object splits.

## Checks without training

```bash
data/env-linux/bin/python LlamaFT/check_labels.py
data/env-linux/bin/python LlamaFT/check_batch.py
```

The label check validates parser/vector command types. The batch check uses the
cached processor on CPU and checks assistant-only labels with both padding
directions; it does not load model weights.

## Longer allocations and restart support

The Slurm scripts request eight hours for new jobs. From the login node:

```bash
mkdir -p logs
sbatch tools/slurm-llama.sbatch LlamaFT/finetune.py --category scene --object_count 3 --nepochs 10
```

Add `--resume_from_checkpoint` to continue the latest saved Trainer checkpoint,
or supply its directory explicitly. Use the same dataset and experiment settings.
`--save_steps` defaults to 25, keeping two checkpoints. The `final/` adapter is
for inference; resuming training requires `checkpoint-N/` with trainer state.
Add `--skip_existing` to inference to skip valid outputs with unchanged settings.
The wrapper sets persistent cache paths and compiler configuration automatically.
New processes still load model weights; no timing improvement is guaranteed.

## Learned placement task

See [placement training and evaluation](../SceneAssembler/PLACEMENT.md) for the new
placement task in both model families. Existing CAD training/inference commands
remain the default. Placement checkpoints are separate and require new training.
