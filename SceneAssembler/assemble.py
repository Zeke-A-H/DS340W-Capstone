#!/usr/bin/env python3
"""Assemble reconstructed object CAD with annotated placement (CPU; no model inference)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 'img2cad.predicted_scene_assembly.v1'


def safe_id(value):
    value = str(value)
    if not value or any(not (c.isalnum() or c in '_-') for c in value):
        raise ValueError(f'Invalid scene/object ID: {value!r}')
    return value


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Expected a finite homogeneous 4x4 transform')
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0).mean()
    if scale <= 0 or np.linalg.det(linear) <= 0 or not np.allclose((linear/scale).T @ (linear/scale), np.eye(3), rtol=1e-6, atol=1e-6):
        raise ValueError('Scene CAD supports positive uniform scale and rigid rotation/translation; shear/reflection is unsupported')
    return matrix


def normalized_to_scene(source_to_scene, source_to_normalized_scale):
    scale = float(source_to_normalized_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Normalization scale must be finite and positive')
    inverse = np.diag([1 / scale, 1 / scale, 1 / scale, 1.])
    return validate_transform(validate_transform(source_to_scene) @ inverse)


def select_scenes(dataset_root, split, object_count=None, scene_ids=None):
    manifest = json.loads((dataset_root / 'manifest.json').read_text())
    scenes = [s for s in manifest['scenes'] if s['split'] == split
              and (object_count is None or s['object_count'] == object_count)]
    if scene_ids:
        requested = set(map(str, scene_ids))
        scenes = [s for s in scenes if str(s['scene_id']) in requested]
        missing = requested - {str(s['scene_id']) for s in scenes}
        if missing:
            raise ValueError(f'Scenes not in requested split/count: {sorted(missing)}')
    if not scenes:
        raise ValueError('No matching scenes')
    return scenes


def prediction_folder(args, entry, object_id, category=None):
    base = args.predictions_dir
    if base is None:
        if args.prediction_mode == 'original':
            base = ROOT/'data/output/baselines/original_on_scenes/trassembler'/category/args.split
        else:
            group = 'all_objects' if args.prediction_mode == 'combined' else f"{entry['object_count']}_objects"
            base = ROOT / 'data/output/trassembler_h5/scene' / group / args.split
    return base / safe_id(object_id)


def source_normalization(metadata, dataset_root, obj):
    """Use the exact scale used by TrAssembler; no centering was applied there."""
    value = metadata.get('reference_source_to_normalized_scale')
    if value is not None:
        return float(value), 'prediction_metadata'
    # Annotated assembly may recover the same scale from the packaged CAD mesh.
    import trimesh
    from cadlib.macro import NORM_FACTOR
    mesh_path = dataset_root / obj['_scene_path'] / Path(obj['cad']).with_name('cad.obj')
    mesh = trimesh.load(str(mesh_path), force='mesh')
    extent = np.max(mesh.bounds[1] - mesh.bounds[0])
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f'Invalid reference CAD bounds: {mesh_path}')
    return float(NORM_FACTOR / extent), 'reference_cad_bounds'


def placement_label(args):
    return 'predicted' if getattr(args, 'placement_source', 'annotated') == 'predicted' else 'ground_truth'


def object_inputs(args, entry, obj):
    object_id = safe_id(obj['object_id'])
    folder = prediction_folder(args, entry, object_id, obj['category'])
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('status') not in ('success', 'skipped'):
        raise ValueError(f"Object reconstruction is not successful: {metadata.get('error', metadata.get('status'))}")
    for key, expected in [('sample_id', object_id), ('scene_id', str(entry['scene_id'])),
                          ('reference_id', str(obj['reference_id'])), ('category', obj['category'])]:
        if str(metadata.get(key)) != str(expected):
            raise ValueError(f'Prediction metadata mismatch: {key}')
    if metadata.get('coordinate_system') != 'normalized_object_local':
        raise ValueError('Prediction coordinate system must be normalized_object_local')
    for name in ('final.step', 'final.h5'):
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    with h5py.File(folder / 'final.h5') as stream:
        vector = stream['vec'][:]
        if vector.ndim != 2 or vector.shape[1] != 17 or not len(vector) or not np.isfinite(vector).all():
            raise ValueError('Invalid predicted CAD vectors')
    if placement_label(args) == 'predicted':
        prediction = args._placement_records.get(object_id)
        if prediction is None or prediction.get('status') != 'success':
            raise ValueError(f'Missing/failed predicted placement for {object_id}')
        from SceneAssembler.placement import SCHEMA as PLACEMENT_SCHEMA
        if prediction.get('schema') != PLACEMENT_SCHEMA:
            raise ValueError('Unknown placement prediction schema')
        if str(prediction.get('scene_id')) != str(entry['scene_id']):
            raise ValueError('Predicted placement scene ID mismatch')
        if prediction.get('coordinate_system') != 'normalized_object_local' or prediction.get('target_coordinate_system') != 'scene':
            raise ValueError('Predicted placement must map normalized_object_local to scene')
        matrix = validate_transform(prediction['normalized_to_scene'])
        scale, scale_source = None, 'not_used_predicted_placement'
        metadata = dict(metadata, placement_checkpoint=prediction.get('checkpoint'),
                        placement_predictions=str(args.placement_predictions.resolve()))
    else:
        scale, scale_source = source_normalization(metadata, args.dataset_root, dict(obj, _scene_path=entry['path']))
        matrix = normalized_to_scene(obj['source_to_scene'], scale)
    return folder, metadata, scale, scale_source, matrix


def read_scene(args, entry):
    scene = json.loads((args.dataset_root / entry['path'] / 'scene.json').read_text())
    if str(scene['scene_id']) != str(entry['scene_id']) or len(scene['objects']) != entry['object_count']:
        raise ValueError('Scene manifest and assembly annotation disagree')
    if scene.get('up_axis', 'Y') != 'Y':
        raise ValueError('This assembly workflow currently expects Y-up scene annotations')
    ids = [safe_id(o['object_id']) for o in scene['objects']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate scene object IDs')
    return scene


def audit_inputs(args, entries):
    results = []
    for entry in entries:
        scene = read_scene(args, entry)
        objects = []
        for obj in scene['objects']:
            item = {'object_id': obj['object_id'], 'ready': False}
            try:
                _, _, scale, provenance, matrix = object_inputs(args, entry, obj)
                item.update(ready=True, source_to_normalized_scale=scale,
                            normalization_source=provenance, normalized_to_scene=matrix.tolist())
            except Exception as error:
                item['error'] = str(error)
            objects.append(item)
        results.append(dict(scene_id=entry['scene_id'], object_count=entry['object_count'],
                            ready=all(o['ready'] for o in objects), objects=objects))
    report = dict(placement=placement_label(args), prediction_mode=args.prediction_mode,
                  ready=all(s['ready'] for s in results), scenes=results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'input_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def load_cad(path):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() == 0:
        raise ValueError(f'Cannot read STEP solid: {path}')
    shape = reader.OneShape()
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError(f'Invalid CAD topology: {path}')
    return shape


def transform_cad(shape, matrix):
    from OCC.Core.gp import gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    matrix = validate_transform(matrix)
    transform = gp_Trsf()
    transform.SetValues(*matrix[:3, :].flatten().tolist())
    operation = BRepBuilderAPI_Transform(shape, transform, True)
    if not operation.IsDone():
        raise ValueError('CAD transformation failed')
    placed = operation.Shape()
    if placed.IsNull() or not BRepCheck_Analyzer(placed).IsValid():
        raise ValueError('Transformed CAD topology is invalid')
    return placed


def write_step(shape, path):
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone or writer.Write(str(path)) != IFSelect_RetDone:
        raise ValueError(f'STEP export failed: {path}')


def write_compound(shapes, path):
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    write_step(compound, path)


def write_obj(meshes, path, placement="ground_truth"):
    offset = 1
    with path.open('w') as stream:
        stream.write(f'# Predicted CAD scene; {placement} placement; Y up; arbitrary units\n')
        for object_id, mesh in meshes:
            stream.write(f'o {object_id}\n')
            for vertex in mesh.vertices:
                stream.write('v ' + ' '.join(f'{v:.10g}' for v in vertex) + '\n')
            for face in mesh.faces:
                stream.write('f ' + ' '.join(str(int(i) + offset) for i in face) + '\n')
            offset += len(mesh.vertices)


def preview_scene(meshes, path, placement="ground_truth"):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection='3d')
    colors = plt.get_cmap('tab10')
    all_points = []
    for index, (object_id, mesh) in enumerate(meshes):
        points = np.asarray(mesh.vertices)[:, [0, 2, 1]]  # Y is vertical in scene coordinates.
        all_points.append(points)
        axis.add_collection3d(Poly3DCollection(points[mesh.faces], facecolor=colors(index % 10),
                                               edgecolor=(0, 0, 0, .12), linewidth=.2, alpha=.95))
    points = np.concatenate(all_points)
    center = (points.min(0) + points.max(0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) / 2, 1e-6) * 1.1
    axis.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
             zlim=(center[2]-radius, center[2]+radius), xlabel='X', ylabel='Z', zlabel='Y',
             title=f'Predicted object geometry · {placement} scene placement')
    axis.set_box_aspect([1, 1, 1])
    axis.view_init(elev=20, azim=35)
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def scene_metrics(meshes, reference_path, seed, placement="ground_truth"):
    import trimesh
    from scipy.spatial import KDTree
    predicted = trimesh.util.concatenate([mesh for _, mesh in meshes])
    reference = trimesh.load(str(reference_path), force='mesh')
    diagonal = float(np.linalg.norm(reference.bounds[1] - reference.bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Invalid reference scene bounds')
    a = trimesh.sample.sample_surface(predicted, 4096, seed=seed)[0] / diagonal
    b = trimesh.sample.sample_surface(reference, 4096, seed=seed+1)[0] / diagonal
    cd = float(KDTree(a).query(b)[0].mean() + KDTree(b).query(a)[0].mean())
    return dict(scene_cd_gt_diagonal=cd, sampled_points_per_mesh=4096,
                definition='Sum of bidirectional mean Euclidean distances in the shared scene frame, divided by reference CAD scene bounding-box diagonal; no independent recentering, scaling, or alignment',
                placement=placement, reference=str(reference_path.resolve()))


def assemble_scene(args, entry, work):
    # CAD imports occur before trimesh/native mesh libraries.
    from utils.io import cad_to_obj
    import trimesh
    scene = read_scene(args, entry)
    source_image = args.dataset_root / entry.get('image', '')
    if source_image.is_file():
        shutil.copyfile(source_image, work / 'input.png')
    shapes, meshes, objects = [], [], []
    with h5py.File(work / 'scene.h5', 'w') as assembly:
        assembly.attrs.update(schema_version=SCHEMA, scene_id=str(entry['scene_id']),
                              placement=placement_label(args), up_axis=scene.get('up_axis', 'Y'), units=scene.get('units', 'arbitrary'))
        group = assembly.create_group('objects')
        for obj in scene['objects']:
            result = {key: obj[key] for key in ('object_id', 'category', 'reference_id')}
            try:
                folder, metadata, scale, provenance, matrix = object_inputs(args, entry, obj)
                placed = transform_cad(load_cad(folder / 'final.step'), matrix)
                local = Path('objects') / safe_id(obj['object_id'])
                destination = work / local
                destination.mkdir(parents=True)
                write_step(placed, destination / 'placed.step')
                cad_to_obj(placed, str(destination / 'placed.obj'))
                mesh = trimesh.load(str(destination / 'placed.obj'), force='mesh')
                if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                    raise ValueError('Empty or non-finite placed mesh')
                item = group.create_group(obj['object_id'])
                item.attrs.update(category=obj['category'], reference_id=str(obj['reference_id']),
                                  cad_coordinate_system='normalized_object_local',
                                  placement=placement_label(args))
                if scale is not None:
                    item.attrs['source_to_normalized_scale'] = scale
                if placement_label(args) == 'ground_truth':
                    item.create_dataset('source_to_scene', data=obj['source_to_scene'])
                item.create_dataset('normalized_to_scene', data=matrix)
                cad = item.create_group('predicted_cad')
                with h5py.File(folder / 'final.h5') as source:
                    for name in source:
                        source.copy(name, cad)
                shapes.append(placed)
                meshes.append((obj['object_id'], mesh))
                result.update(status='success', prediction_dir=str(folder.resolve()),
                              model_checkpoint=metadata.get('model_checkpoint'),
                              input_step_mtime_ns=(folder / 'final.step').stat().st_mtime_ns,
                              source_to_normalized_scale=scale, normalization_source=provenance,
                              source_to_scene=obj['source_to_scene'] if placement_label(args) == 'ground_truth' else None,
                              placement_checkpoint=metadata.get('placement_checkpoint'),
                              placement_predictions=metadata.get('placement_predictions'), normalized_to_scene=matrix.tolist(),
                              placed_step=(local / 'placed.step').as_posix(), placed_obj=(local / 'placed.obj').as_posix())
            except Exception as error:
                if obj['object_id'] in group:
                    del group[obj['object_id']]
                result.update(status='failed', error=str(error))
            objects.append(result)
    complete = len(shapes) == len(scene['objects'])
    report = dict(schema_version=SCHEMA, scene_id=str(entry['scene_id']), split=args.split,
                  prediction_mode=args.prediction_mode, placement=placement_label(args),
                  units=scene.get('units', 'arbitrary'), up_axis=scene.get('up_axis', 'Y'),
                  expected_objects=len(scene['objects']), reconstructed_objects=len(shapes),
                  object_coverage=len(shapes) / len(scene['objects']),
                  status='complete' if complete else ('partial' if shapes and args.allow_partial else 'failed'),
                  source_scene=str((args.dataset_root / entry['path'] / 'scene.json').resolve()), objects=objects)
    if shapes and (complete or args.allow_partial):
        try:
            write_compound(shapes, work / 'scene.step')
            write_obj(meshes, work / 'scene.obj', placement_label(args))
            glb = trimesh.Scene()
            colors = [[76, 120, 168, 255], [245, 133, 24, 255], [84, 162, 75, 255], [228, 87, 86, 255]]
            for index, (object_id, mesh) in enumerate(meshes):
                colored = mesh.copy()
                colored.visual.face_colors = colors[index % len(colors)]
                glb.add_geometry(colored, node_name=object_id, geom_name=object_id)
            glb.export(str(work / 'scene.glb'))
            if not args.no_preview:
                try:
                    preview_scene(meshes, work / 'preview.png', placement_label(args))
                except Exception as error:
                    report['preview_error'] = str(error)
            try:
                seed = (args.seed + int(hashlib.sha256(str(entry['scene_id']).encode()).hexdigest()[:8], 16)) % (2**32-1)
                metrics = scene_metrics(meshes, args.dataset_root / entry['path'] / 'cad.obj', seed, placement_label(args))
                metrics.update(object_coverage=report['object_coverage'], assembly_status=report['status'])
                (work / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
            except Exception as error:
                report['metrics_error'] = str(error)
        except Exception as error:
            report.update(status='failed', assembly_error=str(error))
    with h5py.File(work / 'scene.h5', 'a') as assembly:
        assembly.attrs.update(status=report['status'], expected_objects=report['expected_objects'], reconstructed_objects=report['reconstructed_objects'])
    (work / 'scene.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def run(args):
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir or ROOT / 'data/output/scene_assemblies' / (('predicted/' if placement_label(args) == 'predicted' else '') + args.prediction_mode) / args.split
    if placement_label(args) == 'predicted':
        if not getattr(args, 'placement_predictions', None):
            raise ValueError('--placement_source predicted requires --placement_predictions')
        from SceneAssembler.placement import read_predictions
        rows = read_predictions(args.placement_predictions)
        args._placement_records = {r['sample_id']: r for r in rows}
        if len(args._placement_records) != len(rows):
            raise ValueError('Duplicate object IDs in predicted placements')
    entries = select_scenes(args.dataset_root, args.split, args.object_count, args.scene_id)
    if args.check_inputs:
        report = audit_inputs(args, entries)
        print(f"Ready scenes: {sum(s['ready'] for s in report['scenes'])}/{len(entries)}; {args.output_dir / 'input_audit.json'}")
        return 0 if report['ready'] else 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting before any new scenes are processed.
    for entry in entries:
        destination = args.output_dir / safe_id(entry['scene_id'])
        if destination.exists():
            if not args.overwrite:
                raise FileExistsError(f'{destination}: use --overwrite to replace generated assembly outputs')
            old = json.loads((destination / 'scene.json').read_text())
            if old.get('schema_version') != SCHEMA:
                raise ValueError(f'Refusing to replace an unrecognized output folder: {destination}')
    results = []
    for entry in entries:
        destination = args.output_dir / safe_id(entry['scene_id'])
        work = Path(tempfile.mkdtemp(prefix=f".assembly-{entry['scene_id']}-", dir=args.output_dir))
        try:
            try:
                report = assemble_scene(args, entry, work)
            except Exception as error:
                report = dict(schema_version=SCHEMA, scene_id=str(entry['scene_id']),
                              status='failed', error=str(error), expected_objects=entry['object_count'],
                              reconstructed_objects=0, object_coverage=0, placement=placement_label(args))
                (work / 'scene.json').write_text(json.dumps(report, indent=2) + '\n')
            backup = None
            if destination.exists():
                backup = work.with_name(work.name + '-previous')
                os.replace(destination, backup)
            try:
                os.replace(work, destination)
            except BaseException:
                if backup is not None:
                    os.replace(backup, destination)
                raise
            else:
                if backup is not None:
                    shutil.rmtree(backup)  # Recognized prior assembly, explicit --overwrite only.
        finally:
            if work.exists():
                shutil.rmtree(work)
        results.append(report)
        print(f"{entry['scene_id']}: {report['status']} ({report['reconstructed_objects']}/{report['expected_objects']} objects)", flush=True)
        catalog = []
        for path in sorted(args.output_dir.glob('*/scene.json')):
            item = json.loads(path.read_text())
            if item.get('schema_version') == SCHEMA and path.parent.name == item.get('scene_id'):
                catalog.append(item)
        summary = dict(schema_version=SCHEMA, placement=placement_label(args), split=args.split,
                       prediction_mode=args.prediction_mode, scenes=catalog,
                       complete_scenes=sum(r['status'] == 'complete' for r in catalog),
                       partial_scenes=sum(r['status'] == 'partial' for r in catalog),
                       failed_scenes=sum(r['status'] == 'failed' for r in catalog))
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 1 if any(r['status'] == 'failed' for r in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', type=Path, default=ROOT / 'data/scene_dataset_v2')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--prediction_mode', choices=['combined', 'specialized', 'original'], default='combined')
    parser.add_argument('--placement_source', choices=['annotated','predicted'], default='annotated')
    parser.add_argument('--placement_predictions', type=Path, help='LlamaFT or TrAssembler placement predictions.jsonl')
    parser.add_argument('--predictions_dir', type=Path, help='Override the direct folder containing per-object TrAssembler outputs')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--object_count', type=int, choices=[1, 2, 3, 4])
    parser.add_argument('--scene_id', nargs='+')
    parser.add_argument('--check_inputs', action='store_true')
    parser.add_argument('--allow_partial', action='store_true', help='Export explicitly marked incomplete scenes instead of withholding scene CAD')
    parser.add_argument('--overwrite', action='store_true', help='Replace recognized assembly outputs only; never source predictions')
    parser.add_argument('--no_preview', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    import signal
    def stop_on_termination(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop_on_termination)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
