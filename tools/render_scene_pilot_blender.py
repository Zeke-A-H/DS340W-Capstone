"""Run with Blender: blender -b --python-exit-code 1 --python THIS_FILE -- [options]."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_cad_scenes import ROOT, read_obj, write_png


def material(name, color, emission=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    if emission:
        shader = nodes.new('ShaderNodeEmission')
        shader.inputs['Color'].default_value = (*color, 1)
    else:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value = (*color, 1)
        shader.inputs['Roughness'].default_value = .65
    mat.node_tree.links.new(shader.outputs[0], output.inputs['Surface'])
    return mat


def render_mask(scene, objects, selected, path):
    """White target, black opaque occluders, black world. No lighting dependency."""
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(selected[1] if obj == selected[0] else selected[2])
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    values = np.empty(width*height*4, dtype=np.float32)
    image.pixels.foreach_get(values)
    pixels = values.reshape(height, width, 4)[::-1, :, 0] > .5
    bpy.data.images.remove(image)
    write_png(path, pixels.astype(np.uint8)*255)
    return pixels


def render_scene(source, destination, args):
    annotation = json.loads((source/'annotations.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    if args.device == 'CUDA':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == 'CUDA']
        if not devices:
            raise RuntimeError('No CUDA device visible; use a GPU compute allocation or --device CPU')
        for device in prefs.devices:
            device.use = device.type == 'CUDA'
        scene.cycles.device = 'GPU'
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = args.threads
    width, height = annotation['camera']['image_size']
    if width != height:
        raise ValueError('This renderer currently requires square pilot images')
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (.8, .85, .95, 1)
    background.inputs['Strength'].default_value = .35
    objects = []
    colors = {'chair': (.12, .3, .65), 'table': (.55, .28, .10),
              'storagefurniture': (.16, .45, .23)}
    for record in annotation['objects']:
        vertices, faces = read_obj(ROOT/record['source_mesh'])
        mesh = bpy.data.meshes.new(record['object_id'])
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(record['object_id'], mesh)
        scene.collection.objects.link(obj)
        obj.matrix_world = Matrix(record['source_to_scene'])
        obj.data.materials.append(material(obj.name, colors[record['category']]))
        objects.append(obj)
    camera_info = annotation['camera']
    basis = np.asarray(camera_info['view_basis_columns'])
    right, up, forward = basis[:, 0], basis[:, 1], basis[:, 2]
    center = camera_info['projection_center']
    eye = np.asarray(camera_info['eye']) + right*center[0] + up*center[1]
    camera_data = bpy.data.cameras.new('Camera')
    camera = bpy.data.objects.new('Camera', camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix([[right[i], up[i], -forward[i], eye[i]] for i in range(3)]
                                 + [[0, 0, 0, 1]])
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = width/camera_info['pixels_per_unit']
    scene.camera = camera
    for name, location, power, size in [('Key', (-3, 6, 5), 1100, 5),
                                         ('Fill', (4, 3, -3), 600, 4)]:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.shape, data.size = power, 'DISK', size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, .5, 0))-obj.location).to_track_quat('-Z', 'Y').to_euler()
    # Source coordinates remain Y-up; rotate the Blender plane into XZ.
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, -.005, 0), rotation=(np.pi/2, 0, 0))
    floor = bpy.context.object
    floor.data.materials.append(material('Floor', (.65, .67, .7)))
    if args.background == 'flat-gray':
        floor.hide_render = True
        scene.render.film_transparent = True
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        layers = tree.nodes.new('CompositorNodeRLayers')
        over = tree.nodes.new('CompositorNodeAlphaOver')
        over.inputs[1].default_value = (.35, .35, .35, 1)
        tree.links.new(layers.outputs['Image'], over.inputs[2])
        composite = tree.nodes.new('CompositorNodeComposite')
        tree.links.new(over.outputs['Image'], composite.inputs['Image'])
    scene.view_settings.view_transform = 'AgX'
    scene.render.filepath = str(destination/'image.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(destination/'scene.blend'))
    bpy.ops.render.render(write_still=True)
    # Re-render masks from the same camera. Ground plane and lighting do not
    # contribute; black objects still occlude the white target.
    floor.hide_render = True
    scene.use_nodes = False
    scene.render.film_transparent = False
    background.inputs['Strength'].default_value = 0
    scene.view_settings.view_transform = 'Standard'
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    white = material('MaskWhite', (1, 1, 1), True)
    black = material('MaskBlack', (0, 0, 0), True)
    instances = np.zeros((height, width), dtype=np.uint8)
    for obj, record in zip(objects, annotation['objects']):
        visible = render_mask(scene, objects, (obj, white, black), destination/record['visible_mask'])
        for other in objects:
            other.hide_render = other != obj
        amodal = render_mask(scene, objects, (obj, white, black), destination/record['amodal_mask'])
        for other in objects:
            other.hide_render = False
        assert np.any(visible) and np.all(~visible | amodal), 'Invalid or empty object mask'
        assert not np.any(visible & (instances > 0)), 'Overlapping instance masks'
        instances[visible] = record['instance_id']
        ys, xs = np.where(visible)
        record.update(visible_pixels=int(visible.sum()), amodal_pixels=int(amodal.sum()),
                      visible_bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                      occluded_fraction=1-float(visible.sum())/int(amodal.sum()))
    write_png(destination/'instance_mask.png', instances)
    annotation.update(renderer='Blender '+bpy.app.version_string+' Cycles',
                      mesh=str((source/'scene.obj').relative_to(ROOT)),
                      mesh_path_base='repository root', blend_file='scene.blend',
                      source_annotations=str((source/'annotations.json').relative_to(ROOT)),
                      render_settings={'device': args.device, 'samples': args.samples,
                                       'floor': args.background == 'floor',
                                       'background': args.background, 'color_management': 'AgX',
                                       'mask_threshold': .5})
    (destination/'annotations.json').write_text(json.dumps(annotation, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/cad_scene_pilot')
    parser.add_argument('--output', type=Path, default=ROOT/'data/cad_scene_pilot_blender')
    parser.add_argument('--scene-id', help='Render just this scene for a smoke test')
    parser.add_argument('--device', choices=['CPU', 'CUDA'], default='CUDA')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--background', choices=['flat-gray', 'floor'], default='flat-gray')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.input == args.output or args.input in args.output.parents:
        parser.error('Output must be separate from the input dataset')
    if args.samples < 1 or args.threads < 1:
        parser.error('samples and threads must be positive')
    manifest = json.loads((args.input/'manifest.json').read_text())
    entries = [s for s in manifest['scenes'] if not args.scene_id or s['scene_id'] == args.scene_id]
    if not entries:
        parser.error('No matching scene')
    for entry in entries:
        destination = args.output/entry['path']
        if (destination/'annotations.json').exists():
            print('Skipping completed scene', entry['scene_id'], flush=True)
            continue
        print('Rendering scene', entry['scene_id'], flush=True)
        render_scene(args.input/entry['path'], destination, args)
    # Only publish the entries actually completed in this output directory.
    result = copy.deepcopy(manifest)
    result['scenes'] = [e for e in manifest['scenes'] if (args.output/e['path']/'annotations.json').exists()]
    result['groups'] = {g: sum(e['group'] == g for e in result['scenes']) for g in manifest['groups']}
    result['renderer'] = 'Blender Cycles'
    result['render_status'] = 'blender_complete' if len(result['scenes'])==len(manifest['scenes']) else 'blender_partial'
    if (args.input/'source_split_audit.json').exists():
        shutil.copy2(args.input/'source_split_audit.json', args.output/'source_split_audit.json')
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    for split in ('train', 'val', 'test'):
        (args.output/f'{split}_scene_ids.txt').write_text(''.join(e['scene_id']+'\n' for e in result['scenes'] if e['split']==split))
    print('Finished:', len(result['scenes']), 'completed scenes in', args.output, flush=True)


if __name__ == '__main__':
    main()
