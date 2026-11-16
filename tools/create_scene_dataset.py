"""Build split-safe scene recipes and software previews for Blender rendering."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from create_cad_scenes import ROOT, read_obj, render, write_png
from create_scene_pilot import highest_source_id, load_rgb, COLORS

CATEGORIES = ('chair', 'table', 'storagefurniture')
COUNTS = {'train': 152, 'val': 44, 'test': 20}
COMPOSITIONS = {2: ['table', 'chair'], 3: ['table', 'chair', 'storagefurniture'],
                4: ['table', 'chair', 'chair', 'storagefurniture']}


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def source_pools():
    uses, rows = defaultdict(set), []
    split_hashes = {}
    for category in CATEGORIES:
        mapping = json.loads((ROOT/f'data/anno_id2model_id_{category}.json').read_text())
        for split in COUNTS:
            path = ROOT/f'data/splits/{category}_{split}_ids.txt'
            split_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for object_id in path.read_text().splitlines():
                identity = mapping.get(object_id, category+':'+object_id)
                uses[identity].add(split)
                rows.append((category, split, object_id, identity))
    conflicts = {key: sorted(splits) for key, splits in uses.items() if len(splits)>1}
    pools = defaultdict(list)
    missing = []
    for category, split, object_id, identity in rows:
        if identity not in conflicts:
            folder = ROOT/f'data/raw_annotated/{category}/{object_id}'
            absent = [name for name in ('cad.obj', 'raw.obj', 'cad.h5') if not (folder/name).is_file()]
            if absent:
                missing.append({'category': category, 'reference_id': object_id, 'missing': absent})
                continue
            pools[category, split].append({'reference_id': object_id, 'model_identity': identity})
    return pools, {'excluded_cross_split_model_identities': conflicts,
                   'original_split_sha256': split_hashes, 'excluded_missing_source_files': missing,
                   'safe_pool_sizes': {c+'/'+s: len(pools[c, s]) for c in CATEGORIES for s in COUNTS}}


def next_scene_id():
    maximum = highest_source_id()
    # Only top-level dataset manifests; no recursive mesh traversal.
    for path in (ROOT/'data').glob('*/manifest.json'):
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            continue
        for entry in value.get('scenes', []):
            if str(entry.get('scene_id', '')).isdigit():
                maximum = max(maximum, int(entry['scene_id']))
    return maximum+1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/scene_dataset_v2/_build/recipes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--resume', action='store_true', help='Reproduce the seeded run in an existing output directory')
    parser.add_argument('--start-id', type=int, help='Override the automatically allocated first scene ID')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.resume and args.start_id is None and args.output.exists():
        existing = [int(p.parent.name) for p in args.output.glob('scenes/*/*/annotations.json')
                    if p.parent.name.isdigit()]
        if existing:
            args.start_id = min(existing)
    if args.size < 64:
        parser.error('--size must be at least 64')
    rng = random.Random(args.seed)
    pools, audit = source_pools()
    # Cycled randomized pools limit repeated source models within each split.
    for pool in pools.values():
        rng.shuffle(pool)
    cursors = Counter()
    cache = {}
    first = args.start_id if args.start_id is not None else next_scene_id()
    entries, thumbnails = [], []
    args.output.mkdir(parents=True, exist_ok=args.resume)
    json_write(args.output/'source_split_audit.json', audit)
    def choose(category, split, used):
        pool = pools[category, split]
        for _ in pool:
            item = pool[cursors[category, split] % len(pool)]
            cursors[category, split] += 1
            if item['model_identity'] not in used:
                return item
        raise ValueError('Insufficient distinct source models')
    for split, total in COUNTS.items():
        for local_index in range(total):
            count = local_index % 4+1
            categories = ([CATEGORIES[(local_index//4) % 3]] if count == 1 else COMPOSITIONS[count])
            group = ('table_two_chairs_storagefurniture' if count == 4 else '_'.join(categories))
            scene_id = str(first+len(entries))
            relative = Path('scenes')/f'{count}_object{("s" if count>1 else "")}'/scene_id
            folder = args.output/relative
            folder.mkdir(parents=True, exist_ok=args.resume)
            records, used = [], set()
            source_meshes = []
            for category in categories:
                item = choose(category, split, used)
                used.add(item['model_identity'])
                source = Path(f'data/raw_annotated/{category}/{item["reference_id"]}/cad.obj')
                key = source.as_posix()
                if key not in cache:
                    cache[key] = read_obj(ROOT/source)
                vertices, faces = cache[key]
                source_meshes.append((vertices, faces))
                records.append(dict(item, category=category, source_mesh=key,
                                    source_cad=source.with_name('cad.h5').as_posix(),
                                    source_reference_mesh=source.with_name('raw.obj').as_posix()))
            # Preserve source orientation and category size. Reject intersecting XZ
            # bounding boxes; retry camera-visible layouts instead of hiding objects.
            for attempt in range(100):
                meshes, boxes, transforms = [], [], []
                for (vertices, faces), record in zip(source_meshes, records):
                    low, high = vertices.min(0), vertices.max(0)
                    scale = {'chair': .9, 'table': 1.8, 'storagefurniture': 1.1}[record['category']]
                    scale /= max((high-low)[[0, 2]])
                    linear = np.eye(3)*scale
                    rotated = vertices @ linear.T
                    low, high = rotated.min(0), rotated.max(0)
                    for _ in range(300):
                        x, z = rng.uniform(-2.2, 2.2), rng.uniform(-1.8, 1.8)
                        translation = np.array([x, 0., z])-np.array([(low[0]+high[0])/2, low[1], (low[2]+high[2])/2])
                        box = ((low+translation)[[0, 2]], (high+translation)[[0, 2]])
                        if all(np.any(box[1]+.05 < b[0]) or np.any(b[1]+.05 < box[0]) for b in boxes):
                            break
                    else:
                        break
                    boxes.append(box)
                    transform = np.eye(4)
                    transform[:3, :3], transform[:3, 3] = linear, translation
                    transforms.append(transform)
                    meshes.append((rotated+translation, faces))
                if len(meshes) != count:
                    continue
                # Include a common framing region so repositioning a singleton
                # changes its image position instead of being recentered away.
                anchor = np.array([[x,y,z] for x in (-3.2,3.2)
                                   for y in (0.,2.) for z in (-2.7,2.7)])
                rendered, camera = render(meshes, folder, args.size,
                                          framing_meshes=meshes+[(anchor, np.empty((0,3),dtype=int))],
                                          colors=[COLORS[r['category']] for r in records])
                if all(r['amodal_pixels']>0 and r['visible_pixels']/r['amodal_pixels'] >= .35 for r in rendered):
                    break
            else:
                raise RuntimeError('Unable to place scene '+scene_id)
            offset = 0
            with (folder/'scene.obj').open('w') as obj, (folder/'scene.mtl').open('w') as mtl:
                obj.write('# Y up; arbitrary units\nmtllib scene.mtl\n')
                for instance, (record, mesh, transform, visible) in enumerate(zip(records, meshes, transforms, rendered), 1):
                    record.update(visible, instance_id=instance, object_id=scene_id+'_'+str(instance),
                                  source_to_scene=transform.tolist(),
                                  occluded_fraction=1-visible['visible_pixels']/visible['amodal_pixels'])
                    obj.write(f'o {record["object_id"]}\nusemtl object_{instance}\n')
                    vertices, faces = mesh
                    for v in vertices:
                        obj.write('v '+' '.join(f'{x:.8f}' for x in v)+'\n')
                    for face in faces:
                        obj.write('f '+' '.join(str(int(x)+offset+1) for x in face)+'\n')
                    offset += len(vertices)
                    mtl.write(f'newmtl object_{instance}\nKd '+' '.join(str(v/255) for v in COLORS[record['category']])+'\n\n')
            annotation = {'scene_id': scene_id, 'family_id': scene_id, 'group': group,
                          'object_count': count, 'split': split, 'image': 'image.png',
                          'mesh': 'scene.obj', 'instance_mask': 'instance_mask.png',
                          'units': 'arbitrary', 'up_axis': 'Y', 'camera': camera,
                          'renderer': 'NumPy software preview', 'objects': records}
            json_write(folder/'annotations.json', annotation)
            entries.append({key: annotation[key] for key in ('scene_id','family_id','group','object_count','split')})
            entries[-1]['path'] = relative.as_posix()
            if local_index < 24:
                thumbnails.append(load_rgb(folder/'image.png')[::4, ::4])
            print(f'{split} {local_index+1}/{total}: {scene_id} ({count} objects)', flush=True)
    manifest = {'schema_version': 'img2cad.scene_recipes.v2', 'seed': args.seed,
                'scene_id_range': [first, first+len(entries)-1], 'scenes': entries,
                'groups': dict(Counter(e['group'] for e in entries)),
                'split_counts': COUNTS, 'render_status': 'software_previews_only',
                'source_identity_policy': 'exclude identities present in multiple original splits',
                'pretrained_training_overlap': 'not_verified'}
    manifest['layout_policy'] = 'positions_only; source orientations and per-category sizes preserved'
    json_write(args.output/'manifest.json', manifest)
    (args.output/'splits').mkdir(exist_ok=args.resume)
    for split in COUNTS:
        (args.output/'splits'/f'{split}_scene_ids.txt').write_text(''.join(e['scene_id']+'\n' for e in entries if e['split']==split))
    for split in COUNTS:
        chosen = [e for e in entries if e['split']==split][:24]
        images = [load_rgb(args.output/e['path']/'image.png')[::4, ::4] for e in chosen]
        images += [np.full_like(images[0], 240)]*((-len(images)) % 4)
        write_png(args.output/f'{split}_preview.png', np.concatenate([np.concatenate(images[i:i+4], axis=1)
                                                                    for i in range(0,len(images),4)]))
    (args.output/'README.md').write_text('''# Split-safe scene recipes and previews

216 scenes: 152 train, 44 validation, 20 test. Each split is equally divided
between one, two, three, and four objects. Singletons cycle chair/table/storage
furniture. Other compositions are table+chair, table+chair+storage furniture,
and table+two distinct chairs+storage furniture. Only positions vary with seed
42: source orientations and per-category sizes are fixed. Original splits are unchanged. Identities shared across
original splits are excluded; the audit lists them and hashes the split files.
Source identity means the ShapeNet mapping where available, otherwise category/ID.
This does not establish absence of near-duplicate geometry or checkpoint exposure.

Scenes are grouped physically under scenes/1_object through scenes/4_objects.
Images are SOFTWARE PREVIEWS, not finished Blender renders. Source CAD and raw
geometry paths plus source_to_scene transforms define reference ground truth.
No XZ bounding boxes intersect, and preview visibility is at least 35% per object.
Blender rendering rechecks actual masks. All sizes are approximate, Y is up.
Each preview sheet shows the first 24 scenes of that split (test has 20).

Use tools/slurm-scene-dataset.sbatch to render in Blender with a flat gray
background, package CAD assemblies, and generate padded crop manifests/targets.
The baseline uses ground-truth boxes/categories; detection and model training
are separate work. Training-ready existing LlamaFT text syntax is not generated.
''')


if __name__ == '__main__':
    main()
