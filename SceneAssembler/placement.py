"""Shared placement targets, inputs, transforms and evaluation for both model families.

Inputs use supplied object boxes and calibrated cameras. Targets map normalized
TrAssembler geometry directly to scene coordinates; no reference scale is needed
when applying a predicted transform.
"""
import json
import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'img2cad.normalized_object_placement.v1'
CAMERA_DIM = 20
FEATURE_DIM = 24


def group_name(count):
    return 'all_objects' if count is None else f'{count}_objects'


def decompose(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Expected finite homogeneous 4x4 placement')
    linear = matrix[:3, :3]
    scale = float(np.linalg.norm(linear, axis=0).mean())
    if scale <= 0 or np.linalg.det(linear) <= 0:
        raise ValueError('Placement scale must be positive; reflections are unsupported')
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError('Placement must use rotation and uniform scale')
    return matrix[:3, 3], rotation, scale


def rotation_from_6d(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('rotation_6d must contain six finite numbers')
    first = values[:3]
    if np.linalg.norm(first) < 1e-8:
        raise ValueError('Degenerate rotation axis')
    first = first / np.linalg.norm(first)
    second = values[3:] - np.dot(first, values[3:]) * first
    if np.linalg.norm(second) < 1e-8:
        raise ValueError('Collinear rotation axes')
    second /= np.linalg.norm(second)
    return np.column_stack([first, second, np.cross(first, second)])


def vector_to_matrix(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (10,) or not np.isfinite(vector).all():
        raise ValueError('Placement vector must contain ten finite values')
    if not -20 < vector[9] < 20:
        raise ValueError('Predicted log scale is outside supported numerical range')
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_from_6d(vector[3:9]) * np.exp(vector[9])
    matrix[:3, 3] = vector[:3]
    return matrix


def matrix_to_vector(matrix):
    translation, rotation, scale = decompose(matrix)
    return np.concatenate([translation, rotation[:, 0], rotation[:, 1], [np.log(scale)]])


def camera_features(camera):
    if camera.get('type') != 'orthographic':
        raise ValueError('Placement currently requires the calibrated orthographic dataset camera')
    values = np.concatenate([np.asarray(camera['view_basis_columns']).reshape(-1),
                             camera['projection_center'], [camera['pixels_per_unit']],
                             camera['image_size'], camera['eye'], camera['target']]).astype(np.float32)
    if values.shape != (CAMERA_DIM,) or not np.isfinite(values).all():
        raise ValueError('Invalid camera calibration')
    return values


def load_samples(dataset_root, split, object_count=None, category='scene', include_targets=True):
    root = Path(dataset_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = {str(e['scene_id']): e for e in manifest['scenes'] if e['split'] == split}
    prepared = ROOT / 'data/trassembler_data/scene/all_objects' / f'{split}.json'
    normalization = {}
    if include_targets and prepared.is_file() and root.resolve() == (ROOT / 'data/scene_dataset_v2').resolve():
        normalization = {r['sample_id']: r.get('source_to_normalized_scale') for r in json.loads(prepared.read_text())}
    annotations = {}
    result = []
    for line in (root / f'{split}_objects.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        entry = entries[str(record['scene_id'])]
        if object_count is not None and entry['object_count'] != object_count:
            continue
        if category != 'scene' and record['category'] != category:
            continue
        if record['scene_id'] not in annotations:
            annotations[record['scene_id']] = json.loads((root / record['scene_annotations']).read_text())
        annotation = annotations[record['scene_id']]
        camera = annotation['camera']
        width, height = camera['image_size']
        box = np.asarray(record['crop_box_xyxy'], dtype=np.float32) / [width, height, width, height]
        if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box).all() or not (box <= 1).all() or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError('Invalid object box')
        sample = dict(sample_id=record['sample_id'], scene_id=record['scene_id'], category=record['category'],
                      object_count=entry['object_count'], bbox=box.tolist(), camera=camera,
                      image_path=str(root / entry['image']), crop_path=str(root / record['image']),
                      features=np.concatenate([box, camera_features(camera)]).astype(np.float32))
        if include_targets:
            obj = next(o for o in annotation['objects'] if o['object_id'] == record['sample_id'])
            scale = normalization.get(record['sample_id'])
            if scale is None:
                import trimesh
                from cadlib.macro import NORM_FACTOR
                mesh = trimesh.load(str((root / record['cad_target']).with_name('cad.obj')), force='mesh')
                scale = float(NORM_FACTOR / np.ptp(mesh.vertices, axis=0).max())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid normalization target')
            matrix = np.asarray(obj['source_to_scene']) @ np.diag([1 / scale] * 3 + [1.])
            sample['target'] = matrix_to_vector(matrix).astype(np.float32)
            sample['normalized_to_scene'] = matrix.tolist()
        result.append(sample)
    if not result:
        raise ValueError(f'No placement samples for {split}/{category}/{object_count}')
    return result


def focus_image(sample):
    with Image.open(sample['image_path']) as source:
        image = source.convert('RGB')
    width, height = image.size
    box = np.asarray(sample['bbox']) * [width, height, width, height]
    ImageDraw.Draw(image).rectangle(box.tolist(), outline=(0, 255, 0), width=max(2, width // 160))
    return image


def prompt(sample):
    return ('Predict placement of the object inside the green box in this full scene. '
            'The box and orthographic camera calibration are supplied, not predicted. '
            'Return only JSON with translation (3 scene-coordinate numbers), rotation_6d '
            '(first rotation column followed by the second, 6 numbers), and scale '
            '(one positive number). The transform maps normalized object-local CAD into '
            'the Y-up scene frame in arbitrary units. Do not emit CAD commands. '
            f"Box xyxy normalized: {json.dumps(sample['bbox'])}. "
            f"Camera: {json.dumps(sample['camera'], separators=(',', ':'))}.")


def response(sample):
    vector = sample['target']
    return json.dumps(dict(translation=[round(float(v),6) for v in vector[:3]],
                           rotation_6d=[round(float(v),6) for v in vector[3:9]],
                           scale=round(float(np.exp(vector[9])), 6)), separators=(',', ':'))


def parse_response(text):
    # Accept JSON fences/prose, but never evaluate generated Python.
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
            translation = np.asarray(data['translation'], dtype=float)
            axes = np.asarray(data['rotation_6d'], dtype=float)
            scale = float(data['scale'])
            if translation.shape != (3,) or axes.shape != (6,) or not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid placement JSON')
            matrix = vector_to_matrix(np.concatenate([translation, axes, [np.log(scale)]]))
            return matrix
        except (ValueError, KeyError, TypeError):
            continue
    raise ValueError('No valid placement JSON in model response')


def fit_statistics(samples):
    features = np.stack([s['features'] for s in samples])
    targets = np.stack([s['target'] for s in samples])
    def stats(values):
        deviation = values.std(0)
        return dict(mean=values.mean(0).tolist(), std=np.where(deviation > 1e-6, deviation, 1.).tolist())
    signature = hashlib.sha256(json.dumps([
        dict(sample_id=s['sample_id'], features=s['features'].tolist(), target=s['target'].tolist(),
             image_time=Path(s['image_path']).stat().st_mtime_ns, crop_time=Path(s['crop_path']).stat().st_mtime_ns)
        for s in samples],sort_keys=True).encode()).hexdigest()
    return dict(schema=SCHEMA, data_signature=signature, features=stats(features), translation=stats(targets[:, :3]),
                log_scale=stats(targets[:, 9:10]), training_samples=len(samples),
                rotation_6d_std=targets[:,3:9].std(0).tolist(),
                nonidentity_rotation_samples=int(sum(np.linalg.norm(rotation_from_6d(v[3:9])-np.eye(3))>1e-5 for v in targets)))


def evaluate(predictions, samples, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seen = {}
    for item in predictions:
        key = item['sample_id']
        if key in seen:
            raise ValueError(f'Duplicate placement prediction: {key}')
        seen[key] = item
    expected = {s['sample_id'] for s in samples}
    unknown = set(seen) - expected
    # A full manifest can be evaluated on a filtered subset; do not count those extras.
    rows = []
    for sample in samples:
        row = {k: sample[k] for k in ('sample_id', 'scene_id', 'category', 'object_count')}
        item = seen.get(sample['sample_id'])
        try:
            if item is None or item.get('status') != 'success':
                raise ValueError('Missing/failed placement prediction')
            if item.get('coordinate_system') != 'normalized_object_local' or item.get('target_coordinate_system') != 'scene':
                raise ValueError('Placement coordinate-system mismatch')
            if str(item['scene_id']) != str(sample['scene_id']):
                raise ValueError('Placement scene ID mismatch')
            t, r, scale = decompose(item['normalized_to_scene'])
            gt, gr, gs = decompose(sample['normalized_to_scene'])
            angle = np.degrees(np.arccos(np.clip((np.trace(r.T @ gr) - 1) / 2, -1, 1)))
            row.update(status='success', translation_error=float(np.linalg.norm(t-gt)),
                       rotation_error_degrees=float(angle), scale_relative_error=float(abs(scale-gs)/gs),
                       log_scale_error=float(abs(np.log(scale/gs))))
        except (ValueError, KeyError, TypeError) as error:
            row.update(status='failed', error=str(error))
        rows.append(row)
    def summary(group):
        valid = [r for r in group if r['status'] == 'success']
        result = dict(total=len(group), valid=len(valid), valid_rate=len(valid)/len(group) if group else None)
        for key in ('translation_error', 'rotation_error_degrees', 'scale_relative_error', 'log_scale_error'):
            result['mean_' + key] = float(np.mean([r[key] for r in valid])) if valid else None
        return result
    report = dict(schema=SCHEMA, supplied_localization=True, supplied_camera=True,
                  target_nonidentity_rotations=int(sum(np.linalg.norm(matrix_to_vector(s['normalized_to_scene'])[3:9]-np.array([1,0,0,0,1,0]))>1e-5 for s in samples)),
                  overall=summary(rows), extra_predictions_not_in_selection=len(unknown),
                  by_object_count={str(n): summary([r for r in rows if r['object_count'] == n]) for n in sorted({r['object_count'] for r in rows})},
                  by_category={cat: summary([r for r in rows if r['category'] == cat]) for cat in sorted({r['category'] for r in rows})},
                  definitions=dict(translation='Euclidean distance in scene units', rotation='SO(3) geodesic angle in degrees', scale='Absolute relative error; lower is better; not CAD geometry accuracy'))
    (output / 'placement_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'placement_metrics_per_object.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps(report['overall'], indent=2))
    return report


def read_predictions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
