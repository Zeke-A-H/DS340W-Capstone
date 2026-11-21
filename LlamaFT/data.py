import json
from pathlib import Path

def structure_to_response(structure: dict) -> str:
    """Convert part names and command types into CAD text."""
    blocks = []

    for part in structure["parts"]:
        part_name = part["name"]
        commands = part["command_types"]

        # EOS marks the end of a vector sequence, not a text command.
        command_lines = [
            command for command in commands
            if command != "<EOS>"
        ]

        block = "\n".join([f"# {part_name}", *command_lines])
        blocks.append(block)

    return "\n\n".join(blocks)

def load_scene_samples(category: str, split: str, object_count: int | None = None) -> list[dict]:
    """Load scene-derived object crops and their CAD text responses."""
    if category not in {"chair", "table", "storagefurniture", "scene"}:
        raise ValueError(f"Unknown category: {category}")
    if object_count is not None and object_count not in {1, 2, 3, 4}:
        raise ValueError('object_count must be 1, 2, 3, or 4')

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")

    project_dir = Path(__file__).resolve().parents[1]
    dataset_dir = project_dir / "data" / "scene_dataset_v2"
    manifest_path = dataset_dir / f"{split}_objects.jsonl"
    scene_manifest = json.loads((dataset_dir / 'manifest.json').read_text())
    counts = {entry['scene_id']: entry['object_count'] for entry in scene_manifest['scenes']}

    samples = []

    with manifest_path.open() as manifest:
        for line in manifest:
            if not line.strip():
                continue

            record = json.loads(line)

            if category != 'scene' and record["category"] != category:
                continue
            count = counts[record['scene_id']]
            if object_count is not None and count != object_count:
                continue

            image_path = dataset_dir / record["image"]
            structure_path = dataset_dir / record["structure_target"]

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            structure = json.loads(structure_path.read_text())
            response = structure_to_response(structure)

            samples.append({
                "image_path": str(image_path),
                "response": response,
                "sample_id": record["sample_id"],
                "scene_id": record['scene_id'],
                "category": record['category'],
                "object_count": count,
            })

    if not samples:
        raise ValueError(f"No samples found for {category}/{split}")

    return samples
