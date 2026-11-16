"""Validate packaged scenes, crop targets, geometry and original split separation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from create_cad_scenes import ROOT, read_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    audit = json.loads((root/'source_split_audit.json').read_text())
    for relative, digest in audit['original_split_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, 'Original split changed'
    mappings = {c: json.loads((ROOT/f'data/anno_id2model_id_{c}.json').read_text())
                for c in ('chair','table','storagefurniture')}
    original = {(c,s): set((ROOT/f'data/splits/{c}_{s}_ids.txt').read_text().splitlines())
                for c in mappings for s in ('train','val','test')}
    identities, fingerprints = defaultdict(set), defaultdict(set)
    object_records, counts, minimum_visibility = {}, Counter(), 1.
    scene_ids = set()
    for entry in manifest['scenes']:
        assert entry['scene_id'] not in scene_ids
        scene_ids.add(entry['scene_id'])
        folder = root/entry['path']
        a = json.loads((folder/'annotations.json').read_text())
        assert entry['object_count']==len(a['objects'])
        counts[entry['split'], entry['object_count']] += 1
        with Image.open(folder/a['image']) as im:
            width, height = im.size
        with Image.open(folder/a['instance_mask']) as im:
            instances = np.asarray(im)
        assert instances.shape==(height,width)
        assert set(np.unique(instances)) <= {0} | {o['instance_id'] for o in a['objects']}
        for key in ('mesh','raw_mesh','scene_cad','assembly'):
            assert (folder/a[key]).is_file()
        with h5py.File(folder/a['scene_cad']) as scene:
            assert len(scene['objects'])==len(a['objects'])
            boxes=[]
            for obj in a['objects']:
                category, ref, split = obj['category'], obj['reference_id'], entry['split']
                assert ref in original[category,split]
                identity = mappings[category].get(ref,category+':'+ref)
                assert identity==obj['model_identity']
                assert identity not in audit['excluded_cross_split_model_identities']
                identities[identity].add(split)
                cad = folder/obj['ground_truth']['cad']
                digest=hashlib.sha256(cad.read_bytes()).hexdigest()
                fingerprints[digest].add(split)
                with h5py.File(cad) as original_cad:
                    group=scene['objects'][obj['object_id']]
                    assert set(original_cad)==set(group['cad'])
                    np.testing.assert_array_equal(group['source_to_scene'][:],obj['source_to_scene'])
                with Image.open(folder/obj['visible_mask']) as im: visible=np.asarray(im)>0
                with Image.open(folder/obj['amodal_mask']) as im: amodal=np.asarray(im)>0
                assert visible.any() and np.all(~visible | amodal)
                assert np.array_equal(instances==obj['instance_id'],visible)
                assert int(visible.sum())==obj['visible_pixels']
                assert int(amodal.sum())==obj['amodal_pixels']
                yy,xx=np.where(visible)
                assert obj['visible_bbox_xyxy']==[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
                minimum_visibility=min(minimum_visibility,float(visible.sum())/amodal.sum())
                vertices,_=read_obj(folder/obj['ground_truth']['cad_mesh'])
                transform=np.asarray(obj['source_to_scene'])
                if manifest.get('layout_policy', '').startswith('positions_only'):
                    scale = transform[0,0]
                    assert scale > 0
                    np.testing.assert_allclose(transform[:3,:3], np.eye(3)*scale, atol=1e-10)
                vertices=vertices@transform[:3,:3].T+transform[:3,3]
                assert np.isfinite(vertices).all() and abs(vertices[:,1].min())<1e-6
                box=(vertices.min(0)[[0,2]],vertices.max(0)[[0,2]])
                assert all(np.any(box[1] < b[0]) or np.any(b[1] < box[0]) for b in boxes)
                boxes.append(box)
                object_records[obj['object_id']]=(split,folder,cad)
    assert all(len(s)==1 for s in identities.values()), 'Model identity leakage'
    assert all(len(s)==1 for s in fingerprints.values()), 'Identical CAD file across splits'
    if 'split_counts' in manifest:
        for split, total in manifest['split_counts'].items():
            assert sum(counts[split,n] for n in range(1,5))==total, 'Incomplete scene split'
            assert all(counts[split,n]==total//4 for n in range(1,5)), 'Object-count imbalance'
    crops=Counter()
    for split in ('train','val','test'):
        name = f'{split}_ids.txt' if manifest.get('layout')=='original_style' else f'{split}_scene_ids.txt'
        listed=(root/'splits'/name).read_text().splitlines()
        assert set(listed)=={e['scene_id'] for e in manifest['scenes'] if e['split']==split}
        seen=set()
        for line in (root/f'{split}_objects.jsonl').read_text().splitlines():
            sample=json.loads(line)
            assert sample['sample_id'] not in seen
            seen.add(sample['sample_id'])
            assert object_records[sample['sample_id']][0]==split
            assert (root/sample['cad_target']).resolve()==object_records[sample['sample_id']][2].resolve()
            target=json.loads((root/sample['structure_target']).read_text())
            assert target['parts']
            with Image.open(root/sample['image']) as im:
                assert im.width==im.height and im.width>0
            assert hashlib.sha256((root/sample['image']).read_bytes()).hexdigest()==sample['image_sha256']
            crops[split]+=1
        assert seen=={i for i,(s,_,_) in object_records.items() if s==split}
    report={'validated_scenes':len(scene_ids), 'validated_objects':len(object_records),
            'scene_counts_by_split_and_objects':{s:{str(n):counts[s,n] for n in range(1,5)} for s in ('train','val','test')},
            'crop_counts':dict(crops),'unique_model_identities':len(identities),
            'minimum_visible_fraction':minimum_visibility,
            'canonical_identity_overlap':0,'identical_cad_overlap':0,'original_splits_unchanged':True,
            'pretrained_checkpoint_exposure':'not_verified'}
    (root/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest['validation_status'] = 'passed'
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Compact visual review: columns are object counts; four rows per split.
    sheet = Image.new('RGB',(640,12*180),(225,225,225))
    draw = ImageDraw.Draw(sheet)
    for split_index, split in enumerate(('train','val','test')):
        for count in range(1,5):
            selected = [e for e in manifest['scenes'] if e['split']==split and e['object_count']==count][:4]
            for row, entry in enumerate(selected):
                image = Image.open(root/entry['image']).convert('RGB')
                image.thumbnail((160,160))
                x,y=(count-1)*160,(split_index*4+row)*180
                sheet.paste(image,(x,y))
                draw.text((x+4,y+160),f'{split} {entry["scene_id"]} ({count} obj)',fill=(0,0,0))
                image.close()
    sheet.save(root/'preview.jpg',quality=90)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
