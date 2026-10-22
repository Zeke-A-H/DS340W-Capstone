"""Audit dataset hashes and resume missing files without concurrent request bursts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1] / 'data'
REPO = 'qq456cvb/img2cad-dataset'


def valid(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    digest = hashlib.sha256() if item['sha256'] else hashlib.sha1()
    if not item['sha256']:
        digest.update(f"blob {item['size']}\0".encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == (item['sha256'] or item['blob_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    manifest = ROOT / 'dataset-manifest.json'
    if args.refresh or not manifest.exists():
        info = HfApi().dataset_info(REPO, files_metadata=True)
        manifest.write_text(json.dumps({'revision': info.sha, 'files': [
            {'path': f.rfilename, 'size': f.size,
             'sha256': f.lfs.sha256 if f.lfs else None, 'blob_id': f.blob_id}
            for f in info.siblings]}, indent=2))
    data = json.loads(manifest.read_text())
    groups = defaultdict(lambda: {'total': 0, 'verified': 0, 'bytes': 0, 'verified_bytes': 0})
    pending = []
    for index, item in enumerate(data['files'], 1):
        group = groups[item['path'].split('/')[0]]
        group['total'] += 1
        group['bytes'] += item['size']
        if valid(ROOT / item['path'], item):
            group['verified'] += 1
            group['verified_bytes'] += item['size']
        else:
            pending.append(item)
        if index % 2000 == 0:
            print(f"Checked {index}/{len(data['files'])} files", flush=True)
    report = {'revision': data['revision'], 'groups': dict(groups),
              'pending': [item['path'] for item in pending]}
    (ROOT / 'dataset-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(groups), indent=2), flush=True)
    print(f'{len(pending)} files need downloading. Report: {ROOT / "dataset-audit.json"}', flush=True)
    if not args.download:
        return
    # A single worker with pacing avoids the CLI's parallel HEAD/GET bursts.
    # Retry-After is honored when provided; longer cooldowns handle rate limits.
    pending.sort(key=lambda item: (item['path'].startswith('shapenet_partseg/'), item['path']))
    for index, item in enumerate(pending, 1):
        for attempt in range(8):
            try:
                print(f"[{index}/{len(pending)}] {item['path']}", flush=True)
                path = hf_hub_download(REPO, item['path'], repo_type='dataset',
                                       revision=data['revision'], local_dir=ROOT,
                                       force_download=(ROOT / item['path']).exists())
                if not valid(Path(path), item):
                    raise RuntimeError(f'Checksum mismatch: {path}')
                break
            except Exception as error:
                response = getattr(error, 'response', None)
                status = getattr(response, 'status_code', None)
                if status not in (429, 500, 502, 503, 504) or attempt == 7:
                    raise
                retry = response.headers.get('Retry-After', '0')
                wait = max(float(retry) if retry.isdigit() else 0, min(900, 60 * 2**attempt))
                reset = re.search(r'\bt=(\d+)', response.headers.get('RateLimit', ''))
                if reset:
                    wait = max(wait, int(reset.group(1)) + 1)
                print(f'HTTP {status}; cooling down for {wait:.0f}s', flush=True)
                time.sleep(wait)
        time.sleep(1.0)
    print('All dataset files downloaded and checksum verified.', flush=True)


if __name__ == '__main__':
    main()
