"""Package the Blender pilot into a versioned scene evaluation dataset.

Run with data/env-linux/bin/python (requires h5py and NumPy).
The scene HDF5 is an assembly container, not the existing single-object format.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from create_cad_scenes import ROOT, read_obj


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def write_mesh(path, objects, source_key):
    offset = 0
    with path.open('w') as out:
        out.write('# Scene reference mesh; Y up; arbitrary units\n')
        if source_key == 'source_mesh':
            out.write('mtllib scene.mtl\n')
        for obj in objects:
            vertices, faces = read_obj(ROOT/obj[source_key])
            matrix = np.asarray(obj['source_to_scene'])
            vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            out.write(f'o {obj["object_id"]}\n')
            if source_key == 'source_mesh':
                out.write(f'usemtl object_{obj["instance_id"]}\n')
            for v in vertices:
                out.write('v '+' '.join(f'{x:.8f}' for x in v)+'\n')
            for face in faces:
                out.write('f '+' '.join(str(int(x)+offset+1) for x in face)+'\n')
            offset += len(vertices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/cad_scene_pilot_blender')
    parser.add_argument('--output', type=Path, default=ROOT/'data/scene_dataset')
    parser.add_argument('--allow-previews', action='store_true', help='Package software assets without scene.blend')
    parser.add_argument('--resume', action='store_true', help='Rebuild an interrupted package in the same output folder')
    parser.add_argument('--simple-layout', action='store_true', help='Use raw_annotated, blender_renderings, and original-style split names')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    manifest = json.loads((args.input/'manifest.json').read_text())
    audit = {'source_object_count': 0, 'objects': [], 'usable_texture_objects': 0}
    seen = set()
    entries = []
    renderers = set()
    for entry in manifest['scenes']:
        source = args.input/entry['path']
        annotation = json.loads((source/'annotations.json').read_text())
        renderers.add(annotation['renderer'])
        relative = (Path('raw_annotated')/f'{entry["object_count"]}_object{("s" if entry["object_count"]>1 else "")}'/entry['scene_id']
                    if args.simple_layout else Path(entry['path']))
        destination = args.output/relative
        ground = destination if args.simple_layout else destination/'ground_truth'
        ground.mkdir(parents=True, exist_ok=args.resume)
        for name in ('image.png', 'instance_mask.png', 'scene.blend'):
            if name == 'scene.blend' and args.allow_previews and not (source/name).exists():
                continue
            target = destination/name
            if args.simple_layout and name in ('image.png','instance_mask.png'):
                target = args.output/'blender_renderings'/(entry['scene_id']+('_mask' if name=='instance_mask.png' else '')+'.png')
                target.parent.mkdir(exist_ok=True)
                annotation['image' if name=='image.png' else 'instance_mask'] = os.path.relpath(target,destination)
            shutil.copy2(source/name, target)
        assembly = {'schema_version': 'img2cad.scene_assembly.v1', 'scene_id': entry['scene_id'],
                    'units': 'arbitrary', 'up_axis': 'Y', 'objects': []}
        assembly_h5_name = 'cad.h5' if args.simple_layout else 'scene.h5'
        with h5py.File(ground/assembly_h5_name, 'w') as scene_h5:
            scene_h5.attrs['schema_version'] = assembly['schema_version']
            scene_h5.attrs['scene_id'] = entry['scene_id']
            scene_h5.attrs['up_axis'] = 'Y'
            scene_h5.attrs['units'] = 'arbitrary'
            instances = scene_h5.create_group('objects')
            for obj in annotation['objects']:
                local = Path('objects' if args.simple_layout else 'ground_truth/objects')/obj['object_id']
                object_folder = destination/local
                object_folder.mkdir(parents=True, exist_ok=args.resume)
                for key, name in [('source_cad', 'cad.h5'), ('source_mesh', 'cad.obj'),
                                  ('source_reference_mesh', 'raw.obj')]:
                    shutil.copy2(ROOT/obj[key], object_folder/name)
                for key, name in [('visible_mask', 'visible_mask.png'), ('amodal_mask', 'amodal_mask.png')]:
                    shutil.copy2(source/obj[key], object_folder/name)
                    obj[key] = (local/name).as_posix()
                obj['ground_truth'] = {key: (local/name).as_posix() for key, name in
                                       [('cad', 'cad.h5'), ('cad_mesh', 'cad.obj'), ('raw_mesh', 'raw.obj')]}
                group = instances.create_group(obj['object_id'])
                group.attrs['category'] = obj['category']
                group.attrs['reference_id'] = obj['reference_id']
                group.create_dataset('source_to_scene', data=np.asarray(obj['source_to_scene']))
                with h5py.File(object_folder/'cad.h5', 'r') as original:
                    cad_group = group.create_group('cad')
                    for key in original:
                        original.copy(key, cad_group)
                    for key, value in original.attrs.items():
                        cad_group.attrs[key] = value
                assembly['objects'].append({'object_id': obj['object_id'], 'instance_id': obj['instance_id'],
                                            'category': obj['category'], 'reference_id': obj['reference_id'],
                                            'cad': obj['ground_truth']['cad'],
                                            'source_to_scene': obj['source_to_scene']})
                folder = (ROOT/obj['source_mesh']).parent
                if folder not in seen:
                    seen.add(folder)
                    info = {'source_folder': folder.relative_to(ROOT).as_posix(), 'meshes': {}}
                    for name in ('cad.obj', 'raw.obj'):
                        lines = (folder/name).read_text().splitlines()
                        info['meshes'][name] = {
                            'uv_coordinates': any(l.startswith('vt ') for l in lines),
                            'vertex_colors': any(l.startswith('v ') and len(l.split()) >= 7 for l in lines),
                            'material_references': [l for l in lines if l.startswith('mtllib ')]}
                    info['material_files'] = [p.name for p in folder.glob('*.mtl')]
                    info['texture_images'] = [p.name for p in folder.iterdir()
                                              if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tga')]
                    # These pilot sources have no appearance assets; fail visibly
                    # rather than silently stripping assets on future sources.
                    if info['material_files'] or info['texture_images'] or any(
                            v['uv_coordinates'] or v['vertex_colors'] or v['material_references']
                            for v in info['meshes'].values()):
                        raise RuntimeError('Source appearance assets found: extend packaging/rendering to preserve them: '+str(folder))
                    audit['objects'].append(info)
        write_mesh(ground/'raw.obj', annotation['objects'], 'source_reference_mesh')
        write_mesh(ground/'cad.obj', annotation['objects'], 'source_mesh')
        original_mesh = (ROOT/annotation['mesh'] if annotation.get('mesh_path_base') == 'repository root'
                         else source/annotation['mesh'])
        shutil.copy2(original_mesh.with_suffix('.mtl'), ground/'scene.mtl')
        save_json(ground/'scene.json', assembly)
        annotation.update(schema_version='img2cad.scene_dataset.v1',
                          mesh='cad.obj' if args.simple_layout else 'ground_truth/cad.obj',
                          raw_mesh='raw.obj' if args.simple_layout else 'ground_truth/raw.obj',
                          scene_cad=assembly_h5_name if args.simple_layout else 'ground_truth/scene.h5',
                          assembly='scene.json' if args.simple_layout else 'ground_truth/scene.json', mesh_path_base='scene directory',
                          source_path_base='repository root', training_label_status='not_generated',
                          appearance={'source_textures_available': False,
                                      'current_images': annotation['renderer'],
                                      'future_background': 'flat gray'})
        save_json(destination/'annotations.json', annotation)
        label_folder = args.output/'labels' if args.simple_layout else destination/'labels'
        label_folder.mkdir(exist_ok=args.resume)
        (label_folder/'README.md').write_text('Training responses are not generated yet. The assembly and CAD ground truth\nare available, but scene.h5 is not a drop-in LlamaFT or TrAssembler input.\n')
        output_entry = copy.deepcopy(entry)
        output_entry.update(path=relative.as_posix(),
                            image=(f'blender_renderings/{entry["scene_id"]}.png' if args.simple_layout else entry['path']+'/image.png'),
                            annotations=(relative/'annotations.json').as_posix(), training_label_status='not_generated')
        entries.append(output_entry)
        print('Packaged', entry['scene_id'], flush=True)
    audit['source_object_count'] = len(seen)
    save_json(args.output/'texture_audit.json', audit)
    manifest.update(schema_version='img2cad.scene_dataset.v1', scenes=entries,
                    layout='original_style' if args.simple_layout else 'scene_folders',
                    dataset_role=('training_validation_test' if any(e['split']=='train' for e in entries) else 'evaluation_pilot'),
                    split_policy='source-model grouped; original split membership preserved',
                    pretrained_training_overlap='not_verified', training_labels='not_generated')
    manifest['render_status'] = ('blender_complete' if all(name.startswith('Blender ') for name in renderers)
                                 else 'software_previews_only')
    save_json(args.output/'manifest.json', manifest)
    if (args.input/'source_split_audit.json').exists():
        shutil.copy2(args.input/'source_split_audit.json', args.output/'source_split_audit.json')
    splits = args.output/'splits'
    splits.mkdir(exist_ok=args.resume)
    for split in ('train', 'val', 'test'):
        split_name = f'{split}_ids.txt' if args.simple_layout else f'{split}_scene_ids.txt'
        (splits/split_name).write_text(''.join(e['scene_id']+'\n' for e in entries if e['split']==split))
        if args.simple_layout:
            by_count = splits/'by_object_count'
            by_count.mkdir(exist_ok=True)
            legacy_singleton = by_count/f'1_objects_{split}_ids.txt'
            if legacy_singleton.exists():
                legacy_singleton.unlink()
            for count in range(1,5):
                count_name = f'{count}_object'+('s' if count>1 else '')
                (by_count/f'{count_name}_{split}_ids.txt').write_text(''.join(e['scene_id']+'\n' for e in entries if e['split']==split and e['object_count']==count))
    preview = ROOT/'data/cad_scene_pilot/contact_sheet.png'
    if preview.exists() and not args.simple_layout:
        shutil.copy2(preview, args.output/'layout_preview.png')
    (args.output/'README.md').write_text('''# Scene evaluation dataset

30 scenes in five matched families and six composition groups. All currently
belong to test. Train/validation lists are intentionally empty. Original data
and both pilot directories are unchanged. IDs remain 49535–49564.

Each scenes/<ID>/ contains the Blender image, instance mask, editable scene.blend,
annotations, and ground_truth/. The ground_truth folder contains combined cad.obj
and raw.obj meshes, scene.mtl, scene.json, scene.h5, and objects/<instance ID>/
with the individual cad.h5, cad.obj, raw.obj, visible mask and amodal mask.
Individual geometry stays in source coordinates; source_to_scene transforms
place it into the assembly. Combined meshes have those transforms applied.
All coordinates are Y up in arbitrary units with approximate relative sizes.

scene.h5 uses a NEW assembly schema, img2cad.scene_assembly.v1:
/objects/<object_id>/cad/ holds the unmodified source CAD datasets;
/objects/<object_id>/source_to_scene holds the 4x4 placement matrix.
It is ground truth, not input accepted by existing single-object model loaders.
scene.json lists the assembly and references individual CAD files. Its file paths
and per-scene asset paths are relative to the scene directory. Source provenance
paths are relative to the repository. Manifest paths are relative to this dataset.
Bounding boxes use exclusive maximum pixel coordinates. Instance-mask values
are the local instance_id, with 0 for background. Each instance remains distinct.

labels/ is reserved for future scene-level model responses; there are currently
NO training-ready text responses. CAD extraction, response format, inference,
prediction assembly, and evaluation integration remain future program work.
Pretrained checkpoint training overlap has not been verified. The current test
designation does not prove the models were unseen by published checkpoints.

texture_audit.json records inspection of all 17 source objects: no UVs, vertex
colors, material references, material files, or texture images were present in
their object folders. Current images are copied from the finished Blender run;
they retain its floor/background. Future renders should use a flat gray
background and preserve source textures where available. A rerender is required
to change existing image appearance. layout_preview.png shows the earlier basic
software renders for composition reference, NOT the final Blender images.

To package a new copy, run from the repository root:
data/env-linux/bin/python tools/package_scene_dataset.py --output data/scene_dataset_copy
''')
    if any(e['split']=='train' for e in entries):
        summary = {s: sum(e['split']==s for e in entries) for s in ('train','val','test')}
        (args.output/'README.md').write_text('''# Scene training, validation and test dataset

Split counts: '''+json.dumps(summary)+'''. Scenes are organized under
scenes/1_object, scenes/2_objects, scenes/3_objects, and scenes/4_objects.
Original split files are unchanged. Source models belong to the matching
original split; identities crossing original splits were excluded by the
recipe generator. The provenance audit records original split hashes.

Each scene includes image.png, instance_mask.png, annotations.json, and
ground_truth/{cad.obj,raw.obj,scene.mtl,scene.h5,scene.json}. Individual source
CAD/mesh files and visible/amodal masks live in ground_truth/objects/<ID>/.
Source CAD geometry stays in original coordinates. source_to_scene transforms
apply scale, rotation and translation; combined meshes have these applied.
Units are arbitrary, Y is up. scene.h5 is a NEW assembly container with
/objects/<ID>/cad and /objects/<ID>/source_to_scene. Existing single-object
model loaders cannot read this assembly format directly.

Source provenance paths are relative to the repository. Manifest paths are
relative to this dataset. Annotation and assembly asset paths are relative to
the scene folder. Boxes have exclusive maximum coordinates. Masks use local
instance IDs (0 is background). labels/ is reserved for future whole-scene
responses; these are not existing LlamaFT training text.

Texture inspection is recorded in texture_audit.json. Source appearance assets
are absent for this dataset. Packaging refuses sources with unhandled appearance
assets. Preview packaging may omit scene.blend; renderer and render_settings in
each annotation distinguish software previews from actual Blender images.
Original checkpoint training exposure has not been verified. Near-duplicate
geometry is not ruled out by canonical identity separation.
''')
    if args.simple_layout:
        legacy_preview = args.output/'layout_preview.png'
        if legacy_preview.exists():
            legacy_preview.unlink()
        (args.output/'README.md').write_text('''# Scene dataset v2

216 scenes: 152 training, 44 validation, 20 test. Only object positions vary.
Original orientations and fixed per-category sizes are retained (Y up,
arbitrary units). Scenes with 1/2/3/4 objects are equally represented per split.

raw_annotated/<N>_object[s]/<scene_id>/ contains:
- cad.obj: combined CAD mesh; raw.obj: combined original reference mesh.
- cad.h5: CAD assembly; scene.json: instance list and source-to-scene transforms.
- objects/<instance_id>/: individual cad.h5, cad.obj, raw.obj and masks.
- annotations.json: camera, source IDs, boxes, visibility and asset paths.
- scene.mtl and scene.blend (Blender renders only).

blender_renderings/<scene_id>.png is the input RGB image;
<scene_id>_mask.png contains instance IDs, with 0 for background.
splits/{train,val,test}_ids.txt lists scene IDs, with additional count-specific
lists in splits/by_object_count/. Manifest paths are relative to this dataset.
Per-scene annotation/assembly paths are relative to the scene folder; source
provenance paths are relative to the repository. Boxes have exclusive maxima.

cad.h5 uses a NEW scene assembly schema: /objects/<instance_id>/cad stores
unmodified source CAD datasets, and source_to_scene is the placement matrix.
Existing single-object model loaders need adaptation to read this container.
Original split files are unchanged. Sources follow corresponding original
splits; canonical identities shared across original splits and missing required
files are excluded. source_split_audit.json records exclusions and split hashes.
No claim is made about published checkpoint training exposure or near duplicates.

Images use a flat gray background in Blender; no usable source textures were
found for packaged objects (see texture_audit.json). The _build/ directory is
internal recipes/renders; use the folders above for experiments. Software
previews are labeled in annotations and are not final Blender images.

crops/ and labels/ are generated by prepare_scene_crops.py. Object JSONL lists
link scene crops to original-coordinate CAD targets. Labels enumerate per-part
command types and require loader/prompt integration. Scene-only TrAssembler
training requires reprocessing crops instead of using original-image PKLs.
Whole-scene training responses and automatic detection are not implemented.
''')


if __name__ == '__main__':
    main()
