#!/usr/bin/env python3
"""Evaluate a placement-trained LlamaFT adapter's saved transform predictions."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SceneAssembler.placement import ROOT, group_name, load_samples, read_predictions, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['placement'], default='placement')
    parser.add_argument('--dataset_root', type=Path, default=ROOT/'data/scene_dataset_v2')
    parser.add_argument('--split', choices=['train','val','test'], default='test')
    parser.add_argument('--object_count', type=int, choices=[1,2,3,4])
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    path = args.predictions or ROOT/'data/output/llamaft_placement/scene'/group_name(args.object_count)/args.split/'predictions.jsonl'
    evaluate(read_predictions(path), load_samples(args.dataset_root,args.split,args.object_count), args.output_dir or path.parent)


if __name__ == '__main__':
    main()
