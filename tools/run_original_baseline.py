"""Run the supplied original category models on scene crops and/or isolated tests.

Runs inference/evaluation only. Separate output roots preserve scene-trained work.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES=('chair','table','storagefurniture')


def run(command):
    print('RUN:', ' '.join(map(str,command)),flush=True)
    subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('category',choices=CATEGORIES)
    parser.add_argument('--domain',choices=['scenes','original','both'],default='scenes')
    parser.add_argument('--check_only',action='store_true')
    args=parser.parse_args();category=args.category
    adapter_root=ROOT/'data/ckpts/hf/llamaft'/category
    adapters=[p for p in adapter_root.glob('checkpoint-*') if (p/'adapter_config.json').is_file()]
    if not adapters:
        raise FileNotFoundError(f'Original pretrained adapter missing: {adapter_root}')
    adapter=max(adapters,key=lambda p:int(p.name.split('-')[-1]))
    model=ROOT/'data/ckpts/hf/trassembler'/category
    for path in (model/'last.ckpt',model/f'{category}_mean_scale_stats.json'):
        if not path.is_file():raise FileNotFoundError(path)
    domains=['scenes','original'] if args.domain=='both' else [args.domain]
    counts={}
    for domain in domains:
        if domain=='scenes':
            records=[json.loads(line) for line in (ROOT/'data/scene_dataset_v2/test_objects.jsonl').read_text().splitlines() if line.strip()]
            images=[ROOT/'data/scene_dataset_v2'/r['image'] for r in records if r['category']==category]
        else:
            ids=(ROOT/f'data/splits/{category}_test_ids.txt').read_text().splitlines()
            images=[ROOT/f'data/blender_renderings/{i}.png' for i in ids if i.strip()]
        missing=[str(p) for p in images if not p.is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} missing {domain} images; first: {missing[:3]}')
        counts[domain]=len(images)
    print(json.dumps(dict(category=category,adapter=str(adapter),model=str(model),test_images=counts),indent=2),flush=True)
    if args.check_only:return
    for domain in domains:
        root=ROOT/'data/output/baselines'/f'original_on_{domain}'
        structures=root/'llamaft'/category/'test'
        objects=root/'trassembler'/category/'test'
        run(['LlamaFT/infer.py','--dataset',domain,'--adapter_source','original','--category',category,
             '--adapter_path',adapter,'--split','test','--out_dir',structures,'--save_text','--skip_existing'])
        if domain=='scenes':
            run(['TrAssembler/eval.py','--dataset','scenes','--model_source','original','--category',category,
                 '--model_dir',model,'--predictions_dir',structures,'--output_dir',objects,
                 '--sampler_shift','0.5','--direct_sampling','--eval_partseg'])
        else:
            # The original inference path appends its category to the output root.
            run(['TrAssembler/eval.py','--category',category,'--model_dir',model,'--predictions_dir',structures,
                 '--output_dir',root/'trassembler','--sampler_shift','0.5','--direct_sampling','--skip_processed'])
            isolated=root/'trassembler'/category
            run(['TrAssembler/eval.py','--metrics_only','--category',category,'--model_dir',model,
                 '--output_dir',isolated,'--eval_partseg'])
        (root/category).mkdir(parents=True,exist_ok=True)
        (root/category/'experiment.json').write_text(json.dumps(dict(training='original',test_domain=domain,
             category=category,adapter=str(adapter),geometry_checkpoint=str(model/'last.ckpt'),
             protocol=dict(reference_symmetry=False,reference_arc=False,text_retrieval=False,sampler_shift=.5)),indent=2)+'\n')


if __name__=='__main__':
    main()
