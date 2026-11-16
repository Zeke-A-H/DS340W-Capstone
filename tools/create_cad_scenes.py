"""Create synthetic multi-object CAD mesh scenes using NumPy and the stdlib.

Run: python3 tools/create_cad_scenes.py
Outputs are meshes and basic software renders, not Blender renders or CAD solids.
"""
import argparse
import json
from pathlib import Path
import struct
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = [(80, 140, 210), (210, 140, 75), (100, 175, 120)]


def read_obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'f':
            indices = [int(x.split('/')[0]) for x in fields[1:]]
            indices = [i - 1 if i > 0 else len(vertices) + i for i in indices]
            faces.extend((indices[0], indices[k], indices[k + 1])
                         for k in range(1, len(indices) - 1))
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def write_png(path, pixels):
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width = pixels.shape[:2]
    color_type = 2 if pixels.ndim == 3 else 0
    def chunk(kind, payload):
        return (struct.pack('!I', len(payload)) + kind + payload
                + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff))
    raw = b''.join(b'\0' + row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'
                     + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, color_type, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def render(meshes, folder, size, framing_meshes=None, colors=None):
    # Orthographic camera; source and scene coordinates use Y as up.
    eye = np.array([5., 4., 7.])
    target = np.array([0., .6, 0.])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.stack([right, up, forward], axis=1)
    points = np.concatenate([v for v, _ in (framing_meshes if framing_meshes is not None else meshes)])
    view = (points - eye) @ basis
    center = (view[:, :2].min(0) + view[:, :2].max(0)) / 2
    scale = (size * .84) / np.ptp(view[:, :2], axis=0).max()
    rgb = np.full((size, size, 3), [240, 242, 245], dtype=np.uint8)
    instance = np.zeros((size, size), dtype=np.uint8)
    depth = np.full((size, size), np.inf)
    records = []
    light = np.array([.4, .8, .5])
    light /= np.linalg.norm(light)
    for index, (vertices, faces) in enumerate(meshes, 1):
        projected = (vertices - eye) @ basis
        screen = (projected[:, :2] - center) * scale + size / 2
        screen[:, 1] = size - screen[:, 1]
        amodal = np.zeros((size, size), dtype=np.uint8)
        for face in faces:
            p = screen[face]
            xmin, ymin = np.maximum(np.floor(p.min(0)).astype(int), 0)
            xmax, ymax = np.minimum(np.ceil(p.max(0)).astype(int), size - 1)
            if xmin > xmax or ymin > ymax:
                continue
            a, b, c = p
            denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-10:
                continue
            yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
            xx, yy = xx + .5, yy + .5
            w0 = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / denominator
            w1 = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / denominator
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
            region = np.s_[ymin:ymax+1, xmin:xmax+1]
            amodal[region][inside] = 255
            z = w0*projected[face[0], 2] + w1*projected[face[1], 2] + w2*projected[face[2], 2]
            visible = inside & (z < depth[region])
            normal = np.cross(vertices[face[1]]-vertices[face[0]], vertices[face[2]]-vertices[face[0]])
            shade = .4 + .6 * abs(float(normal @ light)) / max(np.linalg.norm(normal), 1e-12)
            rgb[region][visible] = np.asarray((colors if colors is not None else COLORS)[index-1]) * shade
            depth[region][visible] = z[visible]
            instance[region][visible] = index
        write_png(folder / f'object_{index}_amodal.png', amodal)
        records.append({'amodal_mask': f'object_{index}_amodal.png',
                        'amodal_pixels': int(np.count_nonzero(amodal))})
    write_png(folder / 'image.png', rgb)
    write_png(folder / 'instance_mask.png', instance)
    for index, record in enumerate(records, 1):
        ys, xs = np.where(instance == index)
        record['visible_bbox_xyxy'] = ([int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
                                       if len(xs) else None)
        record['visible_pixels'] = len(xs)
        record['visible_mask'] = f'object_{index}_visible.png'
        write_png(folder / record['visible_mask'], (instance == index).astype(np.uint8)*255)
    return records, {'type': 'orthographic', 'eye': eye.tolist(), 'target': target.tolist(),
                     'view_basis_columns': basis.tolist(), 'projection_center': center.tolist(),
                     'pixels_per_unit': float(scale), 'image_size': [size, size]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/multi_object_eval')
    parser.add_argument('--size', type=int, default=640)
    args = parser.parse_args()
    if args.size < 32:
        parser.error('--size must be at least 32')
    categories = ['chair', 'table', 'storagefurniture']
    ids = {c: (ROOT / f'data/splits/{c}_test_ids.txt').read_text().splitlines() for c in categories}
    layouts = [ [(-1.6, 0), (0, 0), (1.8, 0)],
                [(-1.1, 1.1), (0, -.3), (1.6, -1.3)],
                [(-1.2, -.8), (.2, .6), (1.7, -.7)] ]
    for scene_index, layout in enumerate(layouts):
        folder = args.output / f'scene_{scene_index+1:03d}'
        if folder.exists():
            raise FileExistsError(f'Refusing to overwrite {folder}; choose a new --output directory')
        folder.mkdir(parents=True)
        meshes, annotations = [], []
        with (folder / 'scene.obj').open('w') as output:
            output.write('# CAD mesh scene; Y up; arbitrary units\nmtllib scene.mtl\n')
            offset = 0
            for index, (category, (x, z)) in enumerate(zip(categories, layout), 1):
                object_id = ids[category][scene_index]
                source = Path(f'data/raw_annotated/{category}/{object_id}/cad.obj')
                vertices, faces = read_obj(ROOT / source)
                low, high = vertices.min(0), vertices.max(0)
                scale = [1.0, 2.0, 1.3][index-1] / max(high[0]-low[0], high[2]-low[2])
                translation = np.array([x, 0., z]) - np.array([(low[0]+high[0])/2, low[1], (low[2]+high[2])/2])*scale
                vertices = vertices * scale + translation
                transform = np.eye(4)
                transform[:3, :3] *= scale
                transform[:3, 3] = translation
                output.write(f'o {category}_{object_id}\nusemtl object_{index}\n')
                for v in vertices:
                    output.write('v ' + ' '.join(f'{a:.8f}' for a in v) + '\n')
                for face in faces:
                    output.write('f ' + ' '.join(str(int(a)+offset+1) for a in face) + '\n')
                offset += len(vertices)
                meshes.append((vertices, faces))
                annotations.append({'instance_id': index, 'category': category, 'reference_id': object_id,
                                    'source_mesh': source.as_posix(),
                                    'source_cad': source.with_name('cad.h5').as_posix(),
                                    'source_to_scene': transform.tolist()})
        with (folder / 'scene.mtl').open('w') as output:
            for index, color in enumerate(COLORS, 1):
                output.write(f'newmtl object_{index}\nKd ' + ' '.join(str(c/255) for c in color) + '\n\n')
        records, camera = render(meshes, folder, args.size)
        for annotation, record in zip(annotations, records):
            annotation.update(record)
            assert annotation['visible_pixels'] > 0, 'An object is fully hidden'
        metadata = {'image': 'image.png', 'mesh': 'scene.obj', 'instance_mask': 'instance_mask.png',
                    'split': 'test', 'up_axis': 'Y', 'units': 'arbitrary',
                    'renderer': 'basic NumPy triangle rasterizer', 'camera': camera, 'objects': annotations}
        (folder / 'annotations.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Created {folder}: ' + ', '.join(f'{o["category"]}/{o["reference_id"]}' for o in annotations))


if __name__ == '__main__':
    main()
