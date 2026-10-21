"""Download the published Img2CAD checkpoints without symlinks or duplicate caches."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'ckpts')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info('qq456cvb/img2cad', files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.startswith(('llamaft/', 'trassembler/', 'partseg/'))]
    remaining = sum(f.size or 0 for f in files if not (destination / f.rfilename).is_file() or (destination / f.rfilename).stat().st_size != f.size)
    free = shutil.disk_usage(destination).free
    print(f'Revision: {info.sha}', flush=True)
    print(f'Destination: {destination}', flush=True)
    print(f'Download remaining: {remaining / 2**30:.2f} GiB; free: {free / 2**30:.2f} GiB', flush=True)
    if args.check_only:
        return
    if remaining + 768 * 2**20 > free:
        raise SystemExit('Insufficient disk space (including a 768 MiB reserve). Choose another --destination.')
    manifest = []
    for index, item in enumerate(files, 1):
        print(f'[{index}/{len(files)}] {item.rfilename}', flush=True)
        path = Path(hf_hub_download(info.id, item.rfilename, revision=info.sha, local_dir=destination))
        if path.stat().st_size != item.size:
            raise RuntimeError(f'Incorrect size: {path}')
        lfs = item.lfs
        expected = lfs.sha256 if lfs else None
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 2**20), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        if expected and actual != expected:
            raise RuntimeError(f'SHA256 mismatch: {path}')
        manifest.append({'path': item.rfilename, 'size': item.size, 'sha256': actual})
    (destination / 'download-manifest.json').write_text(json.dumps({'repo': info.id, 'revision': info.sha, 'files': manifest}, indent=2), encoding='utf-8')
    print('All published Img2CAD checkpoint files downloaded and verified.', flush=True)
    print('The separate Llama base model, CLIP, and DINOv2 are not included in this repository.', flush=True)


if __name__ == '__main__':
    main()
