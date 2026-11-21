"""Check real processor batches on CPU without loading model weights or training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor

from LlamaFT import finetune
from LlamaFT.data import load_scene_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['structure','placement'], default='structure')
    parser.add_argument('--processor', type=str, help='Local processor/tokenizer directory')
    args = parser.parse_args()
    path = args.processor
    if path is None:
        snapshots = ROOT/'data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots'
        candidates = sorted(snapshots.glob('*'))
        if not candidates:
            parser.error('No cached processor found; pass --processor /path/to/local/snapshot')
        path = str(candidates[-1])
    finetune.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    finetune.prompt = (ROOT/'LlamaFT/prompt.txt').read_text()
    if args.task == 'placement':
        from SceneAssembler.placement import load_samples, prompt, response
        samples = [dict(s,task='placement',input=prompt(s),response=response(s)) for s in load_samples(ROOT/'data/scene_dataset_v2','train')]
    else:
        samples = load_scene_samples('chair', 'train')
    first = samples[0]
    second = next(s for s in samples[1:] if len(s['response']) != len(first['response']))
    examples = [finetune.format_data(s) for s in (first,second)]
    tokenizer = finetune.processor.tokenizer
    for padding_side in ('right','left'):
        tokenizer.padding_side = padding_side
        batch = finetune.collate_fn(examples)
        assert batch['input_ids'].shape == batch['labels'].shape
        assert (batch['labels'][batch['attention_mask']==0] == -100).all()
        assert (batch['attention_mask']==0).any(), 'Unequal-length examples should exercise padding'
        for i,sample in enumerate((first,second)):
            selected = batch['labels'][i] != -100
            assert selected.any()
            assert torch.equal(batch['labels'][i,selected],batch['input_ids'][i,selected])
            answer = tokenizer.decode(batch['labels'][i,selected],skip_special_tokens=True)
            assert answer.strip()==sample['response'].strip(), (padding_side,repr(answer))
            assert sample.get('input',finetune.prompt) not in answer
        print(f'{padding_side} padding passed: shape={tuple(batch["input_ids"].shape)}, '
              f'supervised tokens={(batch["labels"] != -100).sum(dim=1).tolist()}')
    print('Supervised response example:\n'+first['response'])
    print('Passed. No model weights loaded and no training started.')
    for example in examples:
        example['messages'][0]['content'][1]['image'].close()


if __name__ == '__main__':
    main()
