# SceneAssembler

Assemble per-object TrAssembler predictions into a full CAD scene using the
existing dataset's annotated placement. This is the next stage after LlamaFT
structure prediction and TrAssembler geometry prediction. It runs on **CPU**;
no GPU allocation, model training, or new inference is required.

The default is **predicted object geometry with ground-truth localization and placement**.
A new [learned placement workflow](PLACEMENT.md) supports LlamaFT and TrAssembler
placement training and predicted-transform assembly. It still uses supplied boxes
and camera calibration, not automatic reconstruction from an unseen image. Existing source
predictions and checkpoints are never modified.

## Inputs and coordinate handling

For each scene, the tool reads:

- `data/scene_dataset_v2/manifest.json` and the scene's `scene.json`.
- Object predictions: `final.step`, `final.h5`, and `metadata.json`.
- The exact source-to-normalized scale recorded by scene TrAssembler preparation.
  If older metadata lacks it, the identical scale is recovered from the annotated
  reference CAD mesh bounds and marked `reference_cad_bounds` in the report.

TrAssembler scaled source coordinates by a positive scalar without centering.
The assembly therefore applies:

```text
normalized_to_scene = source_to_scene × inverse(source_to_normalized)
```

Scaling is undone before the annotation's scale, rotation, and translation are
applied. All objects share the annotation's Y-up, arbitrary-unit coordinate frame.
Shear, reflections, nonuniform scaling, invalid metadata, and invalid CAD topology
are rejected. OpenCascade transforms the native CAD geometry, not just its mesh.

## Audit inputs

From the project root, without starting a GPU session:

```bash
data/env-linux/bin/python SceneAssembler/assemble.py --check_inputs
```

The combined model is the default. This checks all 20 held-out scenes (five each
with 1, 2, 3, or 4 objects) and writes `input_audit.json` to the assembly output
root. Scene assembly does not require the missing reference symmetry labels;
those labels affect optional refinement in the previous stage only.

## Build scenes

For a single scene during development:

```bash
data/env-linux/bin/python SceneAssembler/assemble.py --scene_id 49780
```

For all held-out scenes, use a CPU batch allocation rather than a long login-node
process. The script has no GPU request and exits when assembly finishes:

```bash
mkdir -p logs
sbatch --account=read_crch_l1ejh6103 SceneAssembler/assemble.sbatch --overwrite
```

`--overwrite` is needed to replace the example assemblies already generated during
validation. It can replace only recognized SceneAssembler scene folders. It never
clears or overwrites TrAssembler predictions, dataset annotations, or checkpoints.
Without this flag, an existing scene output is an error.

Monitor the batch job:

```bash
tail -f logs/scenes-JOB_ID.log
squeue -j JOB_ID -o "%.18i %.12T %.12M %.12L"
```

To assemble predictions from the count-specific models instead:

```bash
sbatch --account=read_crch_l1ejh6103 SceneAssembler/assemble.sbatch --prediction_mode specialized
```

`combined` uses `data/output/trassembler_h5/scene/all_objects/<split>/`.
`specialized` routes each scene to `<N>_objects/<split>/`. Select a subset with
`--object_count 3`, `--scene_id 49763 49767`, or `--split train|val|test`.
Override a direct per-object input folder using `--predictions_dir`, or an assembly
output root using `--output_dir`. To regenerate any existing selection, add
`--overwrite` explicitly. `--no_preview` skips the CPU-rendered preview.

## Outputs

Run only one assembly job per output root at a time.

Default roots:

```text
data/output/scene_assemblies/combined/test/
data/output/scene_assemblies/specialized/test/
```

Each scene ID has:

```text
<scene_id>/
├── scene.step              # Combined native CAD, retaining separate solids
├── scene.obj               # Combined world-coordinate mesh with object groups
├── scene.glb               # Colored mesh scene with separate named object nodes
├── scene.h5                # Assembly container: object CAD vectors + placement matrices
├── scene.json              # Object identities, transforms, input provenance, status/errors
├── input.png               # Original scene image, when supplied by the manifest
├── preview.png             # Static colored 3D preview, when rendering succeeds
├── metrics.json            # Scene-level geometric comparison
└── objects/<object_id>/
    ├── placed.step         # Individual object CAD in scene coordinates
    └── placed.obj          # Individual object mesh in scene coordinates
```

Original normalized individual CAD files remain in the TrAssembler output folder.
The new placed copies allow inspecting objects independently in their assembled
positions. The scene STEP is a geometric compound of the predicted object solids,
not a fused solid, constraint-based mechanical assembly, or XDE named-component
hierarchy. The JSON/HDF5 and GLB preserve scene object identity explicitly.

`scene.h5` is a scene assembly container, not a single-object `vec` file. Each
object stores its original **normalized** prediction under `predicted_cad/`, and
both `source_to_scene` and `normalized_to_scene` transforms. Do not treat its
CAD vectors as already transformed world-coordinate vectors.

The root `summary.json` catalogs completed, partial, and failed generated scenes.
`input_audit.json` is a prerequisite report and does not certify CAD topology.
A generated file alone does not establish accurate reconstruction; consult the
manifest, metrics, and previews.

## Missing objects and errors

By default, a scene with any missing/failed/invalid object is marked `failed`, and
no whole-scene CAD is exported. Per-object results remain available for diagnosis.
The process exits nonzero if any requested scene fails.

To deliberately export incomplete scenes, add `--allow_partial`. Such scenes are
marked `partial`, include all failed object IDs/errors, and report their object
coverage. A scene with no usable objects still fails. Do not count partial scenes
as complete reconstructions in the paper. Preview/metric errors are recorded
separately and never silently represented as good scores.

## Evaluation

`metrics.json` contains `scene_cd_gt_diagonal`: symmetric mean Euclidean surface
Chamfer distance using 4096 points per mesh, in the shared scene coordinate frame,
divided by the ground-truth CAD scene bounding-box diagonal. **Lower is better.**
No independent recentering, scaling, or registration is applied to predictions.
This differs from the original per-object normalized CD and should be reported as
a distinct scene metric. Finite sampling means identical meshes can have a small
nonzero score. Sampling is seeded for reproducibility.

Report this alongside object coverage and complete-scene success rate. Placement
is supplied by annotations, so these scores do not measure predicted placement.
The comparison target is the packaged `cad.obj`, not the raw textured scene mesh.

Assembly preserves errors already present in reconstructed objects. It cannot
repair distorted parts, missing parts, incorrect proportions, or object geometry
that extends beyond its expected bounds.

## Validation

```bash
data/env-linux/bin/python SceneAssembler/check_assembly.py --check_cad
```

Checks inverse normalization, rotation/translation/scale composition, reference
bounds for all 50 test objects, existing scene artifacts, identity and coverage,
placed mesh consistency, and native STEP topology/solid preservation. Use
`--output_dir` to validate specialized assemblies. Checks are CPU-only; reading
and validating many CAD files can take time, so use a CPU allocation for large runs.

## Comparing original models and scene-trained models

`data/output/comparisons/` contains CSV, PNG, and LaTeX tables generated by:

```bash
data/env-linux/bin/python tools/report_results.py
```

- `category_comparison.*`: shared versus scene-size specialist geometry on the same 50 scene crops, grouped by chair, table, and storagefurniture (cabinet).
- `matched_test_comparison.*`: those two policies on identical objects within each source-scene size.
- `original_model_comparison.*`: original-trained category models on scene crops and, separately, isolated original tests. Unmeasured entries remain blank/dashes.
- `comparison_matches.csv`: scene `sample_id`, scene ID, original `reference_id`, category, count, and corresponding model metrics. For example `49761_1` maps to original chair `2289`. Isolated renderings and scene crops remain different image domains even when the reference CAD identity matches.
- `whole_scene_comparison.*`: combined-model assembly scores by scene size. `whole_scene_model_comparison.csv` also picks up original/specialist assemblies when available; compare equal `scene_id` and placement protocol.
- `comparison_status.json`: source paths, missing baselines, training completion, placement checkpoint availability, and limitations.

Original-model checkpoints and images can be checked without loading GPU models:

```bash
for category in chair table storagefurniture; do
    data/env-linux/bin/python tools/run_original_baseline.py "$category" --domain both --check_only
done
```

To evaluate the supplied original models on exactly the scene test crops:

```bash
for category in chair table storagefurniture; do
    sbatch --account=read_crch_l1ejh6103 tools/slurm-original-baseline.sbatch "$category" --domain scenes
done
```

Use `--domain original` for the separate original isolated test set, or `--domain both` for both domains in each category job. The script requests one V100, 64 GB host RAM, and a three-hour time limit; this is a cap, not a runtime estimate. Adjust `sbatch --mem=... --time=...` if needed. Follow progress with `tail -f logs/original-JOBID.log`. No training is performed. Outputs go to `data/output/baselines/original_on_scenes/` and `original_on_original/`, keeping existing results intact.

After all three scene-baseline jobs finish, assemble their objects and regenerate the tables:

```bash
sbatch --account=read_crch_l1ejh6103 SceneAssembler/assemble.sbatch --prediction_mode original
# After that CPU assembly job finishes:
data/env-linux/bin/python tools/report_results.py
```

The baseline uses direct sampling without reference symmetry/arc assistance or text retrieval, with sampler shift 0.5. Earlier scene prediction metadata does not record the inference options: verify that protocol or regenerate scene predictions before claiming a final controlled paper comparison. Segmentation coverage is reported explicitly. Whole-scene CD uses a shared frame and GT-diagonal normalization; object CD independently centers and normalizes each shape. These two CD columns must not be ranked against each other. Current scene assemblies use annotated placement; placement models have not yet been trained.
