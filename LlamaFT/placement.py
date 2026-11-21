"""Placement inference using the same quantized vision Llama loader as CAD inference."""
import json
import hashlib
from pathlib import Path

from SceneAssembler.placement import (ROOT, SCHEMA, load_samples, focus_image, prompt,
                                      parse_response, group_name, evaluate, read_predictions)


def infer(args):
    from LlamaFT.infer import load_model_and_processor, generate_description
    root = ROOT / 'data/ckpts/llamaft_placement/scene' / group_name(args.object_count)
    adapter = Path(args.adapter_path) if args.adapter_path else root / 'final'
    if not (adapter / 'adapter_config.json').is_file() and args.adapter_path is None:
        checkpoints = sorted(root.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        if checkpoints:
            adapter = checkpoints[-1]
    marker = next((p for p in (adapter/'task_config.json', adapter.parent/'task_config.json') if p.is_file()), None)
    if marker is None or json.loads(marker.read_text()).get('task') != 'placement':
        raise ValueError(f'Expected a placement-trained adapter, not a CAD adapter: {adapter}')
    if not (adapter / 'adapter_config.json').is_file():
        raise FileNotFoundError(adapter/'adapter_config.json')
    output = Path(args.out_dir) if args.out_dir else ROOT / 'data/output/llamaft_placement/scene' / group_name(args.object_count) / args.split
    if output.exists() and any(output.glob('*.h5')):
        raise ValueError('Keep placement outputs separate from existing CAD structure files')
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset_root, args.split, args.object_count, include_targets=False)
    if args.max_samples:
        samples = samples[:args.max_samples]
    weights = adapter/'adapter_model.safetensors'
    if not weights.is_file():
        weights = adapter/'adapter_model.bin'
    checkpoint_signature = f'{adapter.resolve()}:{weights.stat().st_mtime_ns}:{weights.stat().st_size}'
    def signature(sample):
        values = dict(checkpoint=checkpoint_signature, bbox=sample['bbox'], camera=sample['camera'],
                      image=str(Path(sample['image_path']).resolve()), image_time=Path(sample['image_path']).stat().st_mtime_ns)
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
    signatures = {s['sample_id']:signature(s) for s in samples}
    existing = {}
    path = output/'predictions.jsonl'
    if args.skip_existing and path.is_file():
        rows = read_predictions(path)
        existing = {p['sample_id']:p for p in rows}
        if len(existing) != len(rows):
            raise ValueError('Duplicate saved placement prediction IDs')
    reusable = {key:value for key,value in existing.items()
                if value.get('status')=='success' and value.get('input_signature')==signatures.get(key)}
    for value in reusable.values():
        from SceneAssembler.placement import decompose
        decompose(value['normalized_to_scene'])
    model = processor = None
    if any(s['sample_id'] not in reusable for s in samples):
        if args.hf_token:
            from huggingface_hub import login
            login(token=args.hf_token)
        model, processor = load_model_and_processor('meta-llama/Llama-3.2-11B-Vision-Instruct', str(adapter), precision=args.precision)
    predictions = []
    for sample in samples:
        item = {k: sample[k] for k in ('sample_id', 'scene_id', 'category', 'object_count')}
        if sample['sample_id'] in reusable:
            item = dict(reusable[sample['sample_id']], reused=True)
            predictions.append(item)
            (output/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
            print(sample['sample_id'],'reused',flush=True)
            continue
        item['input_signature'] = signatures[sample['sample_id']]
        item.update(schema=SCHEMA, checkpoint=str(adapter.resolve()), coordinate_system='normalized_object_local',
                    target_coordinate_system='scene', supplied_boxes=True, supplied_camera=True)
        try:
            text = generate_description(prompt(sample), focus_image(sample), model, processor,
                                        max_new_tokens=args.num_tokens, do_sample=False)
            item['raw_text'] = text
            matrix = parse_response(text)
            item.update(status='success', normalized_to_scene=matrix.tolist())
            if args.save_text:
                (output / f"{sample['sample_id']}.txt").write_text(text)
        except Exception as error:
            item.update(status='failed', error=str(error))
        predictions.append(item)
        (output/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
        print(sample['sample_id'], item['status'], flush=True)
    targets = load_samples(args.dataset_root, args.split, args.object_count)
    selected = {s['sample_id'] for s in samples}
    evaluate(predictions, [s for s in targets if s['sample_id'] in selected], output)
    if any(p['status'] != 'success' for p in predictions):
        raise SystemExit(1)
