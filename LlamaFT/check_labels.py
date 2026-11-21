"""Check scene response text against the parser used by LlamaFT inference.

Run from the project root:
data/env-linux/bin/python LlamaFT/check_labels.py
No model is loaded and no GPU is needed. The default checks scene-mode
command preservation; --legacy-parser checks the original arc reordering.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LlamaFT.data import structure_to_response
from LlamaFT.infer import get_part_to_cad, get_part_to_vec


def check(path, preserve_command_order=True):
    structure = json.loads(path.read_text())
    expected = {}
    for part in structure['parts']:
        name = part['name']
        if name in expected:
            raise ValueError('Duplicate part name: '+name)
        expected[name] = [c for c in part['command_types'] if c != '<EOS>']
    text = structure_to_response(structure)
    parsed = get_part_to_cad(text.splitlines(), preserve_command_order=preserve_command_order)
    actual = {name: [command[0] for command in commands] for name, commands in parsed.items()}
    if expected != actual:
        names = sorted(set(expected) | set(actual))
        differences = [{ 'part': name, 'expected': expected.get(name), 'parsed': actual.get(name)}
                       for name in names if expected.get(name) != actual.get(name)]
        raise ValueError(json.dumps(differences[:3]))
    # Verify these parsed structures can also be represented as CAD vectors.
    vectors = get_part_to_vec(parsed)
    codes = {'L': 0, 'A': 1, 'R': 2, '<SOL>': 4, 'E': 5}
    for name, commands in expected.items():
        array = vectors[name]
        if array.shape != (len(commands), 17):
            raise ValueError(f'{name}: unexpected vector shape {array.shape}')
        if array[:, 0].tolist() != [codes[c] for c in commands]:
            raise ValueError(f'{name}: vector command types changed')
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT/'data/scene_dataset_v2/labels')
    parser.add_argument('--file', type=Path, help='Check one structure JSON instead of the full directory')
    parser.add_argument('--legacy-parser', action='store_true', help='Check original arc-reordering behavior')
    args = parser.parse_args()
    paths = [args.file] if args.file else sorted(args.labels.glob('*.structure.json'))
    if not paths:
        parser.error('No structure label files found')
    failed, parts = [], 0
    for path in paths:
        try:
            parts += check(path, preserve_command_order=not args.legacy_parser)
        except Exception as error:
            failed.append((path.name, str(error)))
    print(f'Checked {len(paths)} labels: {len(paths)-len(failed)} passed, {len(failed)} failed.')
    if failed:
        for name, reason in failed[:5]:
            print(f'FAIL {name}: {reason}')
        raise SystemExit(1)
    print(f'All {parts} parts preserve names, command order, and vector command types.')


if __name__ == '__main__':
    main()
