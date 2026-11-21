"""CPU checks for normalization/placement math and generated scene artifacts."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from assemble import ROOT, SCHEMA, normalized_to_scene, validate_transform, select_scenes, run


def check_math(dataset_root):
    import trimesh
    angle = .7
    matrix = np.eye(4)
    matrix[:3, :3] = 2 * np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    matrix[:3, 3] = [4, 5, 6]
    points = np.array([[1., 2., 3.], [-2., 1., 4.]])
    combined = normalized_to_scene(matrix, .25)
    assert np.allclose((points * .25) @ combined[:3, :3].T + combined[:3, 3], points @ matrix[:3, :3].T + matrix[:3, 3])
    for invalid in (np.diag([-1., 1., 1., 1.]), np.diag([1., 2., 1., 1.]), np.full((4, 4), np.nan)):
        try:
            validate_transform(invalid)
            raise AssertionError('Invalid CAD transform accepted')
        except ValueError:
            pass
    checked = 0
    for entry in select_scenes(dataset_root, 'test'):
        folder = dataset_root / entry['path']
        reference = trimesh.load(str(folder / 'cad.obj'), force='mesh')
        transformed = []
        for obj in json.loads((folder / 'scene.json').read_text())['objects']:
            source = trimesh.load(str(folder / Path(obj['cad']).with_name('cad.obj')), force='mesh')
            extent = np.ptp(source.vertices, axis=0).max()
            scale = .75 / extent
            transform = np.asarray(obj['source_to_scene'])
            matrix = normalized_to_scene(transform, scale)
            actual = (source.vertices * scale) @ matrix[:3, :3].T + matrix[:3, 3]
            expected = source.vertices @ transform[:3, :3].T + transform[:3, 3]
            assert np.allclose(actual, expected, atol=1e-8)
            transformed.append(actual)
            checked += 1
        vertices = np.concatenate(transformed)
        assert np.allclose(np.stack([vertices.min(0), vertices.max(0)]), reference.bounds, atol=1e-6), entry['scene_id']
    print(f'Normalization inverse, rotation/translation/scale, and reference-scene bounds passed: {checked} objects')


def check_outputs(output, check_cad):
    # Load CAD bindings first when requested.
    if check_cad:
        from assemble import load_cad
    import trimesh
    scenes = sorted(output.glob('*/scene.json'))
    assert scenes, f'No assemblies in {output}'
    counts = {}
    for path in scenes:
        report = json.loads(path.read_text())
        assert report['schema_version'] == SCHEMA
        assert report['status'] == 'complete', (path, report)
        assert report['reconstructed_objects'] == report['expected_objects']
        assert report['placement'] in ('ground_truth','predicted')
        counts[report['expected_objects']] = counts.get(report['expected_objects'], 0) + 1
        folder = path.parent
        for name in ('scene.step', 'scene.obj', 'scene.glb', 'scene.h5', 'metrics.json'):
            assert (folder / name).stat().st_size > 0
        mesh = trimesh.load(str(folder / 'scene.obj'), force='mesh')
        assert len(mesh.faces) and np.isfinite(mesh.vertices).all()
        scene = trimesh.load(str(folder / 'scene.glb'), force='scene')
        assert len(scene.geometry) == report['expected_objects']
        metrics = json.loads((folder / 'metrics.json').read_text())
        assert np.isfinite(metrics['scene_cd_gt_diagonal'])
        reference_scene = json.loads(Path(report['source_scene']).read_text())
        reference_objects = {o['object_id']: o for o in reference_scene['objects']}
        with h5py.File(folder / 'scene.h5') as assembly:
            assert len(assembly['objects']) == report['expected_objects']
            for obj in report['objects']:
                assert obj['status'] == 'success'
                if report['placement'] == 'ground_truth':
                    source_mesh = trimesh.load(str(Path(report['source_scene']).parent / Path(reference_objects[obj['object_id']]['cad']).with_name('cad.obj')), force='mesh')
                    assert np.isclose(obj['source_to_normalized_scale'], .75 / np.ptp(source_mesh.vertices, axis=0).max())
                    matrix = normalized_to_scene(obj['source_to_scene'], obj['source_to_normalized_scale'])
                else:
                    from assemble import validate_transform
                    assert obj['source_to_scene'] is None and obj['source_to_normalized_scale'] is None
                    matrix = validate_transform(obj['normalized_to_scene'])
                assert np.allclose(matrix, obj['normalized_to_scene'])
                assert np.allclose(matrix, assembly['objects'][obj['object_id']]['normalized_to_scene'][:])
                normalized_mesh = trimesh.load(str(Path(obj['prediction_dir']) / 'final.obj'), force='mesh')
                normalized_mesh.apply_transform(matrix)
                placed = trimesh.load(str(folder / obj['placed_obj']), force='mesh')
                # Mesh tessellation may vary with the absolute deflection after scaling.
                tolerance = .005 * max(np.linalg.norm(matrix[:3, :3], axis=0))
                assert np.allclose(normalized_mesh.bounds, placed.bounds, atol=tolerance), obj['object_id']
        if check_cad:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_SOLID
            def solid_count(shape):
                explorer = TopExp_Explorer(shape, TopAbs_SOLID)
                count = 0
                while explorer.More():
                    count += 1
                    explorer.Next()
                return count
            assembled_solids = solid_count(load_cad(folder / 'scene.step'))
            individual_solids = sum(solid_count(load_cad(folder / obj['placed_step'])) for obj in report['objects'])
            assert assembled_solids == individual_solids > 0, 'Objects were lost or fused in STEP assembly'
    print(f'Assembly files, topology/meshes, object identities, and coordinate consistency passed: {len(scenes)} scenes; counts={counts}')


def check_missing_inputs(dataset_root):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        args = SimpleNamespace(dataset_root=dataset_root, split='test', object_count=1, scene_id=['49761'],
                               prediction_mode='combined', predictions_dir=base / 'missing',
                               output_dir=base / 'output', check_inputs=True, allow_partial=False,
                               overwrite=False, no_preview=True, seed=42)
        assert run(args) == 1
        audit = json.loads((args.output_dir / 'input_audit.json').read_text())
        assert not audit['ready'] and not audit['scenes'][0]['objects'][0]['ready']
        assert not (args.output_dir / '49761/scene.step').exists()
    print('Missing predictions are explicitly reported, not silently omitted')


def check_commit_safety():
    """An interrupted overwrite must restore the previous recognized assembly."""
    import os
    import assemble
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        dataset = base / 'dataset'
        dataset.mkdir()
        (dataset / 'manifest.json').write_text(json.dumps({'scenes': [dict(scene_id='7000', split='test', object_count=1, path='unused')]}))
        args = SimpleNamespace(dataset_root=dataset, split='test', object_count=None, scene_id=None,
                               prediction_mode='combined', output_dir=base / 'out', check_inputs=False,
                               overwrite=False, allow_partial=False)
        sequence = []
        def fake_build(args, entry, work):
            sequence.append(1)
            report = dict(schema_version=SCHEMA, scene_id='7000', status='complete',
                          expected_objects=1, reconstructed_objects=1, sequence=len(sequence))
            (work / 'scene.json').write_text(json.dumps(report))
            return report
        with patch.object(assemble, 'assemble_scene', side_effect=fake_build):
            assert run(args) == 0
            original = (args.output_dir / '7000/scene.json').read_bytes()
            args.overwrite = True
            replace = os.replace
            failed = []
            def interrupt_once(source, destination):
                if not failed and Path(source).name.startswith('.assembly-') and Path(destination).name == '7000':
                    failed.append(True)
                    raise RuntimeError('Simulated commit failure')
                return replace(source, destination)
            with patch.object(assemble.os, 'replace', side_effect=interrupt_once):
                try:
                    run(args)
                    raise AssertionError('Commit failure was ignored')
                except RuntimeError:
                    pass
            assert (args.output_dir / '7000/scene.json').read_bytes() == original
            assert run(args) == 0
            assert json.loads((args.output_dir / '7000/scene.json').read_text())['sequence'] == 3
    print('Overwrite replacement and rollback preserve prior scene outputs on failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', type=Path, default=ROOT / 'data/scene_dataset_v2')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'data/output/scene_assemblies/combined/test')
    parser.add_argument('--check_cad', action='store_true', help='Also read and validate every assembled STEP solid')
    args = parser.parse_args()
    check_math(args.dataset_root)
    check_commit_safety()
    check_missing_inputs(args.dataset_root)
    check_outputs(args.output_dir, args.check_cad)


if __name__ == '__main__':
    main()
