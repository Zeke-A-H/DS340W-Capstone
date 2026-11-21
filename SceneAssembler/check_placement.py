"""CPU checks for shared placement targets, metrics, context inputs and assembly routing."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from SceneAssembler.placement import (SCHEMA, load_samples, matrix_to_vector, vector_to_matrix,
                                      parse_response, response, evaluate, fit_statistics, focus_image)


def main():
    all_samples={}
    for split,expected in [('train',380),('val',110),('test',50)]:
        samples=load_samples(ROOT/'data/scene_dataset_v2',split)
        assert len(samples)==expected
        all_samples[split]=samples
        for sample in samples:
            matrix=np.asarray(sample['normalized_to_scene'])
            assert np.allclose(vector_to_matrix(matrix_to_vector(matrix)),matrix,atol=1e-6)
            assert np.allclose(parse_response(response(sample)),matrix,atol=1e-5)
            assert sample['features'].shape==(24,)
        inputs=load_samples(ROOT/'data/scene_dataset_v2',split,include_targets=False)
        assert all('target' not in s and 'normalized_to_scene' not in s for s in inputs)
        assert all(np.array_equal(a['features'],b['features']) for a,b in zip(samples,inputs))
        print(split, len(samples),'placement targets/inputs valid')
    assert not ({s['scene_id'] for s in all_samples['train']} & {s['scene_id'] for s in all_samples['test']})
    stats=fit_statistics(all_samples['train'])
    assert stats['training_samples']==380
    marked=focus_image(all_samples['test'][0]);assert marked.size==(640,640)
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary)
        samples=all_samples['test']
        predictions=[dict(schema=SCHEMA,sample_id=s['sample_id'],scene_id=s['scene_id'],status='success',
                          coordinate_system='normalized_object_local',target_coordinate_system='scene',
                          normalized_to_scene=s['normalized_to_scene']) for s in samples]
        report=evaluate(predictions,samples,folder)
        assert report['overall']['valid_rate']==1
        assert report['overall']['mean_translation_error']==0
        assert report['overall']['mean_scale_relative_error']<1e-12
        assert report['overall']['mean_rotation_error_degrees']<1e-5
        corrupted=[dict(p) for p in predictions]
        corrupted[0]=dict(corrupted[0],normalized_to_scene=np.zeros((4,4)).tolist())
        report=evaluate(corrupted,samples,folder)
        assert report['overall']['valid']==49
        try:evaluate(predictions+[predictions[0]],samples,folder);raise AssertionError('Duplicates accepted')
        except ValueError:pass
        # Change reference placement while leaving pixels/boxes/camera unchanged.
        root=folder/'data';root.mkdir()
        entry=next(s for s in json.loads((ROOT/'data/scene_dataset_v2/manifest.json').read_text())['scenes'] if s['scene_id']==samples[0]['scene_id'])
        record=json.loads((ROOT/'data/scene_dataset_v2/test_objects.jsonl').read_text().splitlines()[0])
        annotation=json.loads((ROOT/'data/scene_dataset_v2'/record['scene_annotations']).read_text())
        original_target=annotation['objects'][0]['source_to_scene'][0][3]
        annotation['objects'][0]['source_to_scene'][0][3]+=100
        (root/'annotation.json').write_text(json.dumps(annotation))
        record['scene_annotations']='annotation.json'
        record['cad_target']=str(ROOT/'data/scene_dataset_v2'/record['cad_target'])
        record['image']=str(ROOT/'data/scene_dataset_v2'/record['image'])
        entry=dict(entry,image=str(ROOT/'data/scene_dataset_v2'/entry['image']))
        (root/'manifest.json').write_text(json.dumps({'scenes':[entry]}))
        (root/'test_objects.jsonl').write_text(json.dumps(record)+'\n')
        changed=load_samples(root,'test',include_targets=False)[0]
        assert np.array_equal(changed['features'],samples[0]['features'])
        assert 'target' not in changed
        supervised=load_samples(root,'test')[0]
        assert np.isclose(supervised['target'][0],original_target+100)
        # A deliberately different predicted translation must be used verbatim by assembly.
        from SceneAssembler.assemble import object_inputs
        scene=json.loads((ROOT/'data/scene_dataset_v2'/entry['path']/'scene.json').read_text())
        prediction=dict(predictions[0]);matrix=np.asarray(prediction['normalized_to_scene']).copy();matrix[0,3]+=7
        prediction['normalized_to_scene']=matrix.tolist()
        args=SimpleNamespace(predictions_dir=ROOT/'data/output/trassembler_h5/scene/all_objects/test',
                             prediction_mode='combined',split='test',placement_source='predicted',
                             placement_predictions=folder/'predictions.jsonl',_placement_records={prediction['sample_id']:prediction},
                             dataset_root=ROOT/'data/scene_dataset_v2')
        _,_,scale,provenance,actual=object_inputs(args,entry,scene['objects'][0])
        assert scale is None and provenance=='not_used_predicted_placement' and np.array_equal(actual,matrix)
        args._placement_records={}
        try:object_inputs(args,entry,scene['objects'][0]);raise AssertionError('Missing predicted pose fell back to GT')
        except ValueError:pass
    rotations=[np.linalg.norm(np.asarray(s['normalized_to_scene'])[:3,:3]/np.linalg.norm(np.asarray(s['normalized_to_scene'])[:3,0])-np.eye(3)) for s in all_samples['train']]
    print('Training objects with nonidentity rotation:',sum(r>1e-5 for r in rotations),'/',len(rotations))
    print('Placement metrics, transform round trips, no target leakage, duplicate/invalid handling, and predicted-assembly routing passed')


if __name__=='__main__':
    main()
