"""Generate box-guided crops and discrete CAD targets for single-object baselines.

Requires Pillow, h5py and NumPy (data/env-linux/bin/python).
These JSON targets describe structure; existing model loaders need adaptation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from create_cad_scenes import ROOT

COMMANDS = {0: 'L', 1: 'A', 2: 'R', 3: '<EOS>', 4: '<SOL>', 5: 'E'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/scene_dataset_v2')
    parser.add_argument('--padding', type=float, default=.1)
    args = parser.parse_args()
    if not 0 <= args.padding <= 1:
        parser.error('padding must be between 0 and 1')
    root = args.dataset.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    lists = {s: [] for s in ('train','val','test')}
    for entry in manifest['scenes']:
        folder = root/entry['path']
        annotation = json.loads((folder/'annotations.json').read_text())
        image = Image.open(folder/annotation['image']).convert('RGB')
        crops = root/'crops'
        crops.mkdir(exist_ok=True)
        labels = root/'labels'
        labels.mkdir(exist_ok=True)
        for obj in annotation['objects']:
            x0, y0, x1, y1 = obj['visible_bbox_xyxy']
            pad = int(np.ceil(max(x1-x0, y1-y0)*args.padding))
            box = [max(0,x0-pad), max(0,y0-pad), min(image.width,x1+pad), min(image.height,y1+pad)]
            cropped = image.crop(box)
            side = max(cropped.size)
            square = Image.new('RGB', (side,side), image.getpixel((0,0)))
            placement = ((side-cropped.width)//2, (side-cropped.height)//2)
            square.paste(cropped, placement)
            crop_path = crops/(obj['object_id']+'.png')
            square.save(crop_path)
            target = {'schema_version': 'img2cad.object_structure.v1', 'parts': []}
            with h5py.File(folder/obj['ground_truth']['cad'], 'r') as cad:
                for part in sorted(cad):
                    if part == 'vec' or 'bbox' in part or not isinstance(cad[part],h5py.Dataset):
                        continue
                    values = cad[part][:]
                    if values.ndim != 2 or values.shape[1] != 17:
                        raise ValueError(f'Unexpected CAD shape {part}: {values.shape}')
                    codes = np.rint(values[:,0]).astype(int)
                    if not np.allclose(values[:,0], codes) or any(int(c) not in COMMANDS for c in codes):
                        raise ValueError('Invalid CAD command codes')
                    target['parts'].append({'name': part, 'command_types': [COMMANDS[int(c)] for c in codes]})
            if not target['parts']:
                raise ValueError('No part command sequences for '+obj['object_id'])
            label_path = labels/(obj['object_id']+'.structure.json')
            label_path.write_text(json.dumps(target, ensure_ascii=False, indent=2)+'\n')
            record = {'sample_id': obj['object_id'], 'scene_id': entry['scene_id'],
                      'split': entry['split'], 'category': obj['category'],
                      'reference_id': obj['reference_id'], 'model_identity': obj.get('model_identity'),
                      'image': crop_path.relative_to(root).as_posix(),
                      'structure_target': label_path.relative_to(root).as_posix(),
                      'cad_target': (folder/obj['ground_truth']['cad']).relative_to(root).as_posix(),
                      'original_image': f'data/blender_renderings/{obj["reference_id"]}.png',
                      'original_preprocessed_sample': f'data/trassembler_data/{obj["category"]}_pkl/{obj["reference_id"]}.pkl',
                      'scene_annotations': (folder/'annotations.json').relative_to(root).as_posix(),
                      'crop_box_xyxy': box, 'square_padding_xy': list(placement),
                      'crop_size': side, 'occluded_fraction': obj['occluded_fraction'],
                      'localization': 'ground_truth_box_and_category',
                      'image_sha256': hashlib.sha256(crop_path.read_bytes()).hexdigest()}
            lists[entry['split']].append(record)
            record['original_preprocessed_available'] = (ROOT/record['original_preprocessed_sample']).is_file()
        image.close()
    for split, records in lists.items():
        (root/f'{split}_objects.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    report = {'scene_counts': dict(Counter(e['split'] for e in manifest['scenes'])),
              'crop_counts': {s: len(r) for s,r in lists.items()},
              'localization': 'ground_truth boxes and categories',
              'training_integration': 'JSON targets require loader/prompt adaptation; no training run',
              'trassembler_integration': 'source PKLs contain original image features; rebuild with scene crops',
              'original_paths_base': 'repository root', 'crop_and_target_paths_base': 'dataset root'}
    (root/'crop_report.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest['object_structure_targets'] = 'generated_json_command_types'
    manifest['whole_scene_training_responses'] = 'not_generated'
    manifest['existing_model_training_compatibility'] = 'loader_and_preprocessing_adaptation_required'
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    readme = root/'README.md'
    readme.write_text(readme.read_text().split('\n## Object-crop baseline data')[0])
    with readme.open('a') as out:
        out.write('''\n## Object-crop baseline data

train_objects.jsonl, val_objects.jsonl and test_objects.jsonl index padded
square RGB crops and their CAD/structure targets. Crops include the scene
context and occluders; no mask removes other objects. Boxes/categories are
ground truth. Structure targets enumerate each part's CAD command types;
they do not supply the existing LlamaFT text syntax, extrusion operation
semantics, or a complete executable CAD program. Continuous parameters remain
in cad_target. Loader/prompt integration is required before model training.
Scene-level multi-object training responses are still not generated.

Original-image and PKL paths are provenance/comparison references, not scene
training samples. Existing PKLs contain original-image tensors: do not train
on them unchanged when claiming scene-only training. Rebuild preprocessing
using the provided crop paths. crop_report.json records the sample counts.
''')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
