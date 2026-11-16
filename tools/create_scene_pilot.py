"""Generate five matched families across six composition groups (30 scenes)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from create_cad_scenes import ROOT, read_obj, render, write_png


GROUPS = {
    'chair': [0],
    'storagefurniture': [3],
    'table': [2],
    'table_chair': [2, 0],
    'table_chair_storagefurniture': [2, 0, 3],
    'table_two_chairs_storagefurniture': [2, 0, 1, 3],
}
COLORS = {'chair': (80, 140, 210), 'table': (210, 140, 75),
          'storagefurniture': (100, 175, 120)}


def highest_source_id():
    """Include raw folders, split IDs, renderings, and mapping annotation keys."""
    ids = set()
    for category in ('chair', 'table', 'storagefurniture'):
        ids.update(int(p.name) for p in (ROOT / 'data/raw_annotated' / category).iterdir()
                   if p.name.isdigit())
    for path in (ROOT / 'data/splits').glob('*_ids.txt'):
        ids.update(int(s) for s in path.read_text().splitlines() if s.isdigit())
    for path in (ROOT / 'data/blender_renderings').iterdir():
        stem = path.stem[:-5] if path.stem.endswith('_mask') else path.stem
        if stem.isdigit():
            ids.add(int(stem))
    for path in (ROOT / 'data').glob('anno_id2model_id_*.json'):
        ids.update(int(k) for k in json.loads(path.read_text()) if k.isdigit())
    return max(ids)


def load_rgb(path):
    # Read our writer's RGB PNGs, whose scanlines always use filter zero.
    data = path.read_bytes()
    pos, chunks = 8, []
    while pos < len(data):
        length = struct.unpack('!I', data[pos:pos+4])[0]
        kind, payload = data[pos+4:pos+8], data[pos+8:pos+8+length]
        if kind == b'IHDR':
            width, height = struct.unpack('!II', payload[:8])
        if kind == b'IDAT':
            chunks.append(payload)
        pos += 12 + length
    rows = np.frombuffer(zlib.decompress(b''.join(chunks)), dtype=np.uint8).reshape(height, width*3+1)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/cad_scene_pilot')
    parser.add_argument('--size', type=int, default=640)
    args = parser.parse_args()
    if args.size < 64:
        parser.error('--size must be at least 64')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    maximum = highest_source_id()
    source_ids = {c: sorted((ROOT / f'data/splits/{c}_test_ids.txt').read_text().splitlines(), key=int)
                  for c in ('chair', 'table', 'storagefurniture')}
    entries, sheet_rows = [], []
    for family in range(5):
        # Families alternate identical and distinct chair models.
        chair_a = source_ids['chair'][2*family]
        chair_b = chair_a if family % 2 == 0 else source_ids['chair'][2*family+1]
        specs = [('chair', chair_a, -1.5, .8),
                 ('chair', chair_b, .1, 1.65),
                 ('table', source_ids['table'][family], 0., -.15),
                 ('storagefurniture', source_ids['storagefurniture'][family], 1.8, -1.1)]
        family_meshes, sources = [], []
        for slot, (category, object_id, x, z) in enumerate(specs):
            source = Path(f'data/raw_annotated/{category}/{object_id}/cad.obj')
            vertices, faces = read_obj(ROOT / source)
            low, high = vertices.min(0), vertices.max(0)
            scale = {'chair': .9, 'table': 2., 'storagefurniture': 1.3}[category] / max(high[0]-low[0], high[2]-low[2])
            transform = np.eye(4)
            transform[:3, :3] *= scale
            transform[:3, 3] = np.array([x, 0., z]) - scale*np.array([(low[0]+high[0])/2, low[1], (low[2]+high[2])/2])
            family_meshes.append((vertices*scale + transform[:3, 3], faces))
            sources.append({'family_instance_id': f'family_{family+1:02d}_slot_{slot+1}',
                            'category': category, 'reference_id': object_id,
                            'source_mesh': source.as_posix(),
                            'source_cad': source.with_name('cad.h5').as_posix(),
                            'source_reference_mesh': source.with_name('raw.obj').as_posix(),
                            'source_to_scene': transform.tolist()})
        thumbnails = []
        for group, slots in GROUPS.items():
            scene_id = str(maximum + len(entries) + 1)
            relative = Path('scenes') / scene_id
            folder = args.output / relative
            folder.mkdir(parents=True)
            meshes = [family_meshes[s] for s in slots]
            colors = [COLORS[sources[s]['category']] for s in slots]
            records, camera = render(meshes, folder, args.size, framing_meshes=family_meshes, colors=colors)
            objects = []
            with (folder / 'scene.obj').open('w') as obj, (folder / 'scene.mtl').open('w') as mtl:
                obj.write('# Y up; arbitrary units\nmtllib scene.mtl\n')
                offset = 0
                for instance, (slot, mesh, record, color) in enumerate(zip(slots, meshes, records, colors), 1):
                    annotation = dict(sources[slot], **record, instance_id=instance,
                                      object_id=f'{scene_id}_{instance}')
                    assert 0 < record['visible_pixels'] <= record['amodal_pixels']
                    annotation['occluded_fraction'] = 1-record['visible_pixels']/record['amodal_pixels']
                    objects.append(annotation)
                    obj.write(f'o {annotation["object_id"]}\nusemtl object_{instance}\n')
                    vertices, faces = mesh
                    for vertex in vertices:
                        obj.write('v ' + ' '.join(f'{v:.8f}' for v in vertex) + '\n')
                    for face in faces:
                        obj.write('f ' + ' '.join(str(int(v)+offset+1) for v in face) + '\n')
                    offset += len(vertices)
                    mtl.write(f'newmtl object_{instance}\nKd ' + ' '.join(str(v/255) for v in color) + '\n\n')
            annotation = {'scene_id': scene_id, 'family_id': f'family_{family+1:02d}',
                          'group': group, 'split': 'test', 'image': 'image.png',
                          'mesh': 'scene.obj', 'instance_mask': 'instance_mask.png',
                          'units': 'arbitrary', 'up_axis': 'Y', 'camera': camera,
                          'renderer': 'NumPy software rasterizer', 'objects': objects}
            (folder / 'annotations.json').write_text(json.dumps(annotation, indent=2)+'\n')
            entry = {k: annotation[k] for k in ('scene_id', 'family_id', 'group', 'split')}
            entry.update(path=relative.as_posix(), object_count=len(objects))
            entries.append(entry)
            thumbnails.append(load_rgb(folder/'image.png')[::4, ::4])
            print(f'{scene_id}: {group}, family {family+1}', flush=True)
        sheet_rows.append(np.concatenate(thumbnails, axis=1))
    manifest = {'source_maximum_id': maximum, 'scene_id_range': [maximum+1, maximum+30],
                'groups': {k: 5 for k in GROUPS}, 'scenes': entries}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'test_scene_ids.txt').write_text('\n'.join(e['scene_id'] for e in entries)+'\n')
    write_png(args.output/'contact_sheet.png', np.concatenate(sheet_rows, axis=0))
    (args.output/'README.md').write_text('''# CAD scene pilot

30 scenes: five matched families, six groups per family. All source objects
come from the current test splits. This is an evaluation-only subset.

Scene IDs start one above the maximum numeric source ID found in raw object
folders, split lists, rendering filenames, and annotation mapping keys.
Original object IDs are retained as reference_id in each scene annotation.
The manifest records the maximum, ID range, group and family membership.

contact_sheet.png: rows are families 1–5; columns are chair, storage furniture,
table, table+chair, table+chair+storage furniture, table+two chairs+storage furniture.
Camera, framing, source geometry, colors, and placement remain fixed within a
family. Families 1, 3, 5 repeat the chair model; families 2, 4 use distinct chairs.
Each chair still has a unique scene object ID and persistent family instance ID.

Each scenes/<ID>/ contains image.png, scene.obj, scene.mtl, annotations.json,
instance_mask.png, and object_N_visible.png / object_N_amodal.png masks.
Instance masks use local instance IDs (0 is background); these are not category
IDs. Visible boxes use exclusive maximum pixel coordinates. Amodal masks show
the isolated silhouette. Occlusion fractions compare visible and amodal pixels.
Source paths are relative to the repository root. Scene paths are relative to
this folder; per-scene annotation paths are relative to the scene folder.

Coordinates use Y up, approximate sizes, and arbitrary units. 4x4 transforms
map source column vectors to scene coordinates. Source CAD files are unchanged.
Combined OBJ files are triangle meshes, not merged parametric CAD solids.
Images use basic orthographic software rendering without textures or shadows.
This is a workflow pilot, not a photorealistic benchmark. No model inference
has been run, and original-ID splits have not been altered. Additional source
identity deduplication would be needed before designing new training splits.

Regenerate into a new folder with:
python3 tools/create_scene_pilot.py --output data/another_scene_pilot

Crop generation, model input integration, prediction assembly and scoring are
future pipeline work. These assets supply the inputs and ground truth.
''')
    assert Counter(e['group'] for e in entries) == Counter({k: 5 for k in GROUPS})


if __name__ == '__main__':
    main()
