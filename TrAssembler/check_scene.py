"""CPU validation of prepared scene crops, training statistics and vector decoding."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset import SceneCADDataset, ROOT, load_records, decode_vectors
from dataset import collate_fn


def check_evaluation_paths():
    """Exercise actual metric functions and CLI routing without allocating a GPU."""
    import argparse
    import ast
    import contextlib
    import io
    import sys
    import tempfile
    from types import SimpleNamespace
    import trimesh
    from scipy.spatial import KDTree
    from tqdm import tqdm

    source = ast.parse(Path(__file__).with_name('eval.py').read_text())
    names = {'center_data_single', 'normalize_data_single', 'chamfer_distance_kdtree',
             'compute_cd', 'scene_refinement_inputs', 'scene_records', 'summarize_scene_metrics', 'compute_scene_metrics', 'main'}
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np, torch=torch, json=json, Path=Path, ROOT=ROOT,
                     trimesh=trimesh, KDTree=KDTree, tqdm=tqdm, argparse=argparse, h5py=h5py)
    from dataset import group_name
    namespace['group_name'] = group_name
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'scene_eval_checks', 'exec'), namespace)
    mesh = trimesh.creation.box()
    first = namespace['compute_cd'](mesh, mesh, seed=123)
    assert first == namespace['compute_cd'](mesh, mesh, seed=123), 'CD sampling is not reproducible'
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        output = base / 'out'
        output.mkdir()
        mesh.export(base / 'raw.obj')
        records = []
        for number in range(3):
            sample = str(number)
            records.append(dict(sample_id=sample, scene_id='scene', category='chair',
                                object_count=number + 1, cad_target='cad.h5'))
            folder = output / sample
            folder.mkdir()
            if number < 2:
                mesh.export(folder / 'final.obj')
            else:
                # An obsolete OBJ must not make a failed reconstruction count as success.
                mesh.export(folder / 'final.obj')
                (folder / 'metadata.json').write_text(json.dumps({'status': 'failed', 'error': 'test failure'}))
        records[1]['cad_target'] = 'missing/cad.h5'
        namespace['load_records'] = lambda *args: records
        args = SimpleNamespace(output_dir=output, dataset_root=base, split='test',
                               object_count=None, max_samples=None, seed=42,
                               eval_partseg=False, data_root=str(base))
        with contextlib.redirect_stdout(io.StringIO()):
            report = namespace['compute_scene_metrics'](args)
        assert report['overall']['total_samples'] == 3
        assert report['overall']['successful_generations'] == 2
        assert report['overall']['valid_cd_count'] == 1
        assert np.isclose(report['overall']['mean_occ'], 2 / 3)
        assert set(report['by_object_count']) == {'1', '2', '3'}
        assert (output / 'metrics_per_object.csv').is_file()
        assert len((output / 'metrics_per_object.jsonl').read_text().splitlines()) == 3
        json.loads((output / 'metrics.json').read_text())
        args.eval_partseg = True
        args.partseg_ckpt_dir = str(base / 'checkpoints')
        args.partseg_device = 'cpu'
        def fake_partseg(**kwargs):
            assert [r['sample_id'] for r in kwargs['records']] == ['0', '1']
            return dict(num_samples=1, num_requested=2, accuracy=.5, instance_avg_iou=.25,
                        per_sample=[dict(sample_id='0', accuracy=.5, instance_avg_iou=.25)])
        namespace['evaluate_partseg'] = fake_partseg
        with contextlib.redirect_stdout(io.StringIO()):
            segmented = namespace['compute_scene_metrics'](args)
        assert segmented['overall']['partseg_sample_count'] == 1
        assert segmented['overall']['partseg_accuracy'] == .5
        def missing_partseg(**kwargs):
            raise FileNotFoundError('missing test checkpoint')
        namespace['evaluate_partseg'] = missing_partseg
        with contextlib.redirect_stdout(io.StringIO()):
            missing = namespace['compute_scene_metrics'](args)
        assert missing['part_segmentation']['chair']['status'] == 'unavailable'
        assert missing['overall']['partseg_accuracy'] is None
        # Repeated scene instances must retain their identities while sharing reference annotations.
        from utils.partseg import PartSegDataset
        annotations = base / 'shapenet_partseg' / 'chairs'
        annotations.mkdir(parents=True)
        (annotations.parent / 'synsetoffset2category.txt').write_text('Chair chairs\n')
        (base / 'anno_id2model_id_chair.json').write_text(json.dumps({'source': 'annotated'}))
        np.savetxt(annotations / 'annotated.txt', np.array([[0, 0, 0, 12], [1, 0, 0, 12], [0, 1, 0, 12], [0, 0, 1, 12]]))
        instance_records = [dict(sample_id=str(i), reference_id='source') for i in range(3)]
        # This record has no predicted mesh; it must be excluded rather than replaced with noise.
        instance_records.append(dict(sample_id='missing', reference_id='source'))
        dataset = PartSegDataset(output, 'Chair', n_points=64, data_root=base, records=instance_records, seed=42)
        assert dataset.valid_ids == ['0', '1', '2']
        assert dataset.anno_ids == ['annotated'] * 3
        assert dataset[0][3] == '0' and dataset[1][3] == '1'
        assert np.array_equal(dataset[0][0], dataset[0][0]), 'Scene part sampling is not reproducible'

    calls = []
    namespace['compute_scene_metrics'] = lambda args: calls.append(('metrics', args))
    namespace['infer_scenes'] = lambda args: calls.append(('infer', args))
    namespace['compute_metrics_only'] = lambda args, cat: calls.append(('original', args))
    previous = sys.argv
    try:
        for flags, mode in [(['--dataset', 'scenes', '--metrics_only', '--eval_partseg'], 'metrics'),
                            (['--dataset', 'scenes', '--original_eval', '--save_intermediate',
                              '--symmetry_refinement', '--use_gt_arc_args'], 'infer'),
                            (['--metrics_only', '--category', 'chair'], 'original')]:
            sys.argv = ['eval.py', *flags]
            namespace['main']()
            assert calls[-1][0] == mode
            if '--original_eval' in flags:
                assert all(getattr(calls[-1][1], key) for key in ('text_emb_retrieval', 'symmetry_refinement', 'use_gt_arc_args'))
    finally:
        sys.argv = previous
    refinement_args = SimpleNamespace(dataset_root=ROOT / 'data/scene_dataset_v2',
                                      sym_labels_dir=ROOT / 'data/sym_labels',
                                      symmetry_refinement=True, use_gt_arc_args=True)
    from dataset import load_records
    checked = 0
    missing_symmetry = 0
    for record in load_records(refinement_args.dataset_root, 'test'):
        path = Path(refinement_args.sym_labels_dir) / f"{record['reference_id']}.txt"
        if path.is_file():
            symmetry, arc = namespace['scene_refinement_inputs'](record, refinement_args)
        else:
            missing_symmetry += 1
            try:
                namespace['scene_refinement_inputs'](record, refinement_args)
                raise AssertionError('Missing symmetry label was silently accepted')
            except FileNotFoundError:
                pass
            arc_args = SimpleNamespace(**vars(refinement_args))
            arc_args.symmetry_refinement = False
            symmetry, arc = namespace['scene_refinement_inputs'](record, arc_args)
        assert symmetry in ('rotational', 'reflection', 'no_symmetry')
        with h5py.File(refinement_args.dataset_root / record['cad_target']) as stream:
            arcs = [stream[name][:][stream[name][:, 0] == 1, 1:]
                    for name in sorted(stream) if name != 'vec' and 'bbox' not in name and isinstance(stream[name], h5py.Dataset)]
        expected = next((values[0] for values in arcs if len(values)), None)
        assert (arc is None and expected is None) or np.array_equal(arc, expected)
        checked += 1
    print(f'Reference arc routing checked for {checked} crops; {missing_symmetry} missing symmetry labels correctly rejected')
    print('Scene metrics, failure accounting, deterministic CD, output files, and original/scene CLI parity checks passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared_dir', type=Path, default=ROOT / 'data/trassembler_data/scene/all_objects')
    parser.add_argument('--dataset_root', type=Path, default=ROOT / 'data/scene_dataset_v2')
    args = parser.parse_args()
    torch.set_num_threads(2)
    all_ids = set()
    moments = {}
    for split in ('train', 'val', 'test'):
        dataset = SceneCADDataset(args.prepared_dir, split)
        ids = {r['sample_id'] for r in dataset.records}
        assert len(ids) == len(dataset) and not (all_ids & ids), 'Duplicate IDs or split overlap'
        all_ids.update(ids)
        examples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            record = dataset.records[i]
            assert sample['data_id'] == record['sample_id']
            assert sample['img_tex'].shape == (3, 400, 400)
            assert np.isfinite(sample['img_tex']).all()
            assert sample['img_tex'].min() >= -1 and sample['img_tex'].max() <= 1
            assert record['source_to_normalized_scale'] > 0
            assert sample['part_name_emb'].shape == (len(sample['part_cad']), 512)
            assert np.allclose(np.linalg.norm(sample['part_name_emb'], axis=-1), 1, atol=1e-3)
            with h5py.File(args.dataset_root / record['cad_target']) as stream:
                names = sorted(k for k in stream if k != 'vec' and 'bbox' not in k)
                assert len(names) == len(sample['part_cad'])
                for name, part in zip(names, sample['part_cad']):
                    target = stream[name][:]
                    assert np.array_equal(part[:, 0], target[target[:, 0] != 3, 0]), 'Command order changed'
            batch = collate_fn([deepcopy(sample)])
            decoded = decode_vectors(batch, batch['args'])
            assert len(decoded) == len(sample['part_cad'])
            for original, restored in zip(sample['part_cad'], decoded):
                assert np.array_equal(original[:, 0], restored[:, 0])
                # Collation deliberately fixes extrusion flags/size and arc direction.
                for command, columns in ((0, [1, 2]), (1, [1, 2, 3]), (2, [1, 2, 5]), (5, list(range(6, 12)) + [13, 14])):
                    rows = np.isclose(original[:, 0], command)
                    assert np.allclose(original[rows][:, columns], restored[rows][:, columns], atol=1e-4)
            if split == 'train':
                for j in range(batch['args'].shape[-1]):
                    values = batch['args'][..., j][batch['args_mask'][..., j] > 0].numpy().astype(np.float64)
                    if values.size:
                        moments.setdefault(j, []).append(values)
            if len(examples) < 2:
                examples.append(sample)
        collated = collate_fn(examples)
        assert collated['args'].shape[0] == len(examples)
        assert np.isfinite(collated['args'].numpy()).all()
        print(f'{split}: {len(dataset)} valid samples; command order and vector round-trip passed')
    stats = json.loads((args.prepared_dir / 'mean_scale_stats.json').read_text())
    for j, chunks in moments.items():
        values = np.concatenate(chunks)
        assert np.isclose(stats[str(j)]['mean'], values.mean(), atol=1e-6)
        assert np.isclose(stats[str(j)]['std'], max(values.std(), 1e-4), atol=1e-6)
    for split in ('train', 'val', 'test'):
        combined = load_records(args.dataset_root, split)
        grouped = [r for count in range(1, 5) for r in load_records(args.dataset_root, split, count)]
        assert {r['sample_id'] for r in combined} == {r['sample_id'] for r in grouped}
    print('Training-only normalization statistics and count filtering passed')
    check_evaluation_paths()


if __name__ == '__main__':
    main()
