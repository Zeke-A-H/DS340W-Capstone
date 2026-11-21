import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import numpy as np
from cadlib.macro import *
from einops import rearrange
# Load CAD DLLs before dataset imports OpenCV's bundled native libraries.
from cadlib.visualize import vec2CADsolid
from dataset import collate_fn, group_name, load_records, crop_tensor, decode_vectors, ROOT
from utils.io import cad_to_obj
from tqdm import tqdm
import clip
from functools import partial
import h5py
import os
from omegaconf import OmegaConf
from model import GMFlowModel
import cv2
from scipy.spatial import KDTree
import trimesh
from pathlib import Path
import json
import sys
from torchvision.transforms import ToTensor, Normalize, Compose
from einops import rearrange
from functools import partial
from dataset import extract_base_name, remove_leading_ending_symbols, replace_digits_with_zero
from glob import glob
import argparse
from utils.misc import render_nvdiffrast_rgb
from utils.diffsdf import param2mesh_torch, param2sdf
import logging
from scipy.spatial.transform import Rotation
from utils.partseg import evaluate_partseg

torch.set_grad_enabled(False)
# Set the trimesh logger to suppress everything below ERROR
trimesh.util.attach_to_log(level=logging.ERROR)

def center_data_single(pc):
    centroid = np.mean(pc, axis=0)
    pc[:, 0] -= centroid[0]
    pc[:, 1] -= centroid[1]
    pc[:, 2] -= centroid[2]
    return pc


def create_sample(h5, part_names, data_id, image, do_retrieval, name_embeddings_db, clip_model, group_by_base=False):
    part_cad_vec = []
    for name in part_names:
        part_vec = h5[name]
        while type(part_vec) == h5py._hl.group.Group:
            for k, v in part_vec.items():
                part_vec = v
                break
        part_cad_vec.append(part_vec[:])

    base_part_names = [extract_base_name(remove_leading_ending_symbols(part.replace('#', ' '))) for part in part_names]

    part_name_embeddings = get_clip_features(clip_model, base_part_names, 'cuda').cpu().numpy()  # [n_part, n_dim]

    if do_retrieval:
        # find nearest name in db
        for i in range(len(part_name_embeddings)):
            dist = np.linalg.norm(part_name_embeddings[i] - name_embeddings_db, axis=-1)
            nearest_idx = dist.argmin()
            part_name_embeddings[i] = name_embeddings_db[nearest_idx]
            # part_names[i] = name_db[nearest_idx]
            
    part_names_no_digit = np.array(base_part_names if group_by_base else [replace_digits_with_zero(n) for n in part_names])
    part_index = np.zeros(len(part_names))
    for name in np.unique(part_names_no_digit):
        group_idx = np.where(part_names_no_digit == name)[0]
        part_index[group_idx] = np.arange(len(group_idx))


    sample = {
        "part_cad": part_cad_vec,
        "part_name_emb": part_name_embeddings,
        "part_index": part_index,
        "img_tex": image,
        "img_non_tex": image,
        "data_id": data_id,
    }
    
    return sample


def normalize_data_single(pc):
    # get furthest point distance then normalize
    d = max(np.sum(np.abs(pc)**2, axis=-1)**(1./2))
    pc /= d

    return pc


def chamfer_distance_kdtree(pc1, pc2):
    """
    Compute the Chamfer Distance between two point clouds using KD-trees.

    Parameters:
    - pc1: Numpy array of shape (N, D) representing the first point cloud.
    - pc2: Numpy array of shape (M, D) representing the second point cloud.

    Returns:
    - distance: The Chamfer Distance between the two point clouds.
    """
    # Create KD-trees for each point cloud
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)

    # For each point in pc1, find its nearest neighbor in pc2
    distances1, _ = tree1.query(pc2)
    # For each point in pc2, find its nearest neighbor in pc1
    distances2, _ = tree2.query(pc1)

    # The Chamfer Distance is the sum of both components
    return np.mean(distances1) + np.mean(distances2)


def compute_cd(mesh1, mesh2, normalize=True, seed=None):
    n_points = 2048
    # Suppress trimesh sampling messages
    pts1 = trimesh.sample.sample_surface_even(mesh1, n_points, **({'seed': seed} if seed is not None else {}))[0]
    pts2 = trimesh.sample.sample_surface_even(mesh2, n_points, **({'seed': seed + 1} if seed is not None else {}))[0]

    # if number of points is less than 2048, randomly sample points for remaining points
    if len(pts1) < n_points:
        pts1 = np.concatenate([pts1, (trimesh.sample.sample_surface(mesh1, n_points - len(pts1), seed=seed + 2)[0] if seed is not None else mesh1.sample(n_points - len(pts1)))])
    
    if len(pts2) < n_points:
        pts2 = np.concatenate([pts2, (trimesh.sample.sample_surface(mesh2, n_points - len(pts2), seed=seed + 3)[0] if seed is not None else mesh2.sample(n_points - len(pts2)))])
    
    if normalize:
        pts1 = center_data_single(pts1)
        pts1 = normalize_data_single(pts1)
        pts2 = center_data_single(pts2)
        pts2 = normalize_data_single(pts2)
    cd = chamfer_distance_kdtree(pts1, pts2)
    return cd


def symmetry_opt(gmflow, x_t, t, cmd, cmd_mask, args_mask, gt_arc_arg, model: GMFlowModel, sym_type: str):
    # print(t)
    
    if sym_type == 'no_symmetry':
        return x_t
    
    num_iter = int(np.sqrt(1000 - t.item())) - 20
    if num_iter <= 0:
        return x_t
    has_arc = False
    x_t = rearrange(x_t.clone(), 'B N P L -> B P L N')
    model.denormalize(x_t, args_mask)
    x_t = x_t[0].clone()
    cmd = cmd[0].clone()
    cmd_mask = cmd_mask[0].clone()
    # print(x_t.shape, cmd.shape, cmd_mask.shape, t)
    p_indices, l_indices = torch.nonzero(cmd == 5, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 0:5] = -1
        x_t[p_indices, l_indices, -3:] = 0
        x_t[p_indices, l_indices, -5] = 1.
    p_indices, l_indices = torch.nonzero(cmd == 4, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices] = -1
    p_indices, l_indices = torch.nonzero(cmd == 0, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:] = -1
    p_indices, l_indices = torch.nonzero(cmd == 1, as_tuple=True)
    if p_indices.shape[0] > 0:
        has_arc = True
        x_t[p_indices, l_indices, 4:] = -1
        if gt_arc_arg is not None:
            x_t[p_indices, l_indices, 3] = gt_arc_arg[3]
        else:
            x_t[p_indices, l_indices, 3] = 1.
        x_t[p_indices, l_indices, 2] *= 180 / np.pi
    p_indices, l_indices = torch.nonzero(cmd == 2, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:4] = -1
        x_t[p_indices, l_indices, 5:] = -1
    cmd[cmd_mask == 0] = -1

    x_t = x_t.detach().requires_grad_(True)
    P, L, N = x_t.shape
    optimizer = optim.Adam([x_t], lr=1e-4)
    # bbox_min, bbox_max = mesh.bounds
    # bbox_min = torch.from_numpy(bbox_min).cuda()
    # bbox_max = torch.from_numpy(bbox_max).cuda()
    # part_vecs_torch = part_vecs_torch.clone().detach().requires_grad_(True)
    size = torch.ones((P, 2), dtype=torch.float32, device='cuda', requires_grad=False)
    loc = torch.zeros((P, 3), dtype=torch.float32, device='cuda', requires_grad=False)
    
    # bbox_min = torch.full((3,), -0.5, dtype=torch.float32, device='cuda', requires_grad=False)
    # bbox_max = torch.full((3,), 0.5, dtype=torch.float32, device='cuda', requires_grad=False)

    with torch.enable_grad():
        # Run optimization for 100 iterations
        for iter in range(num_iter):
            part_vecs = torch.cat([cmd[..., None], x_t], -1)

            # Compute SDF for original and reflected points
            verts, faces = param2mesh_torch([None] * P, part_vecs, size, loc, rad=False)
            if iter == 0:
                bbox_min = torch.min(verts, dim=0)[0].detach()
                bbox_max = torch.max(verts, dim=0)[0].detach()
            
            # Sample query points with x < 0 within the bounding box
            query_points = torch.rand((1024, 3), device='cuda') * (bbox_max - bbox_min) + bbox_min
            query_points = query_points[query_points[:, 0] < 0]  # Keep only points with x < 0
            
            sdf_original = param2sdf([None] * P, part_vecs, size, loc, query_points, rad=False, verts=verts, faces=faces)
            if sym_type == 'reflection':
                            # Reflect the query points along the x=0 plane
                reflected_points = query_points.clone()
                reflected_points = reflected_points * torch.tensor([-1.,1,1], device=reflected_points.device)
                sdf_reflected = param2sdf([None] * P, part_vecs, size, loc, reflected_points, rad=False, verts=verts, faces=faces)
                symmetry_loss = torch.mean(torch.abs(sdf_original - sdf_reflected))
            elif sym_type == 'rotational':
                bins = 10
                symmetry_loss = 0.
                for i in range(bins):
                    rot_around_y = Rotation.from_euler('y', 360 * i / bins, degrees=True).as_matrix()
                    rotated_points = query_points.clone()
                    rotation = torch.as_tensor(rot_around_y, dtype=rotated_points.dtype, device=rotated_points.device)
                    rotated_points = rotated_points @ rotation.T
                    sdf_reflected = param2sdf([None] * P, part_vecs, size, loc, rotated_points, rad=False, verts=verts, faces=faces)
                    symmetry_loss += torch.mean(torch.abs(sdf_original - sdf_reflected))
                symmetry_loss /= bins
            else:
                raise ValueError(f"Invalid symmetry type: {sym_type}")
            # Compute symmetry loss as the mean absolute difference between the two SDFs
            # print(f"Iteration {iter}, Symmetry Loss: {symmetry_loss.item()}")

            # Backpropagation and optimization step
            optimizer.zero_grad()
            symmetry_loss.backward()
            if has_arc:  # no optimization for arc
                p_indices, l_indices = torch.nonzero(cmd == 1, as_tuple=True)
                x_t.grad[p_indices, l_indices, 2] = 0.
            optimizer.step()
            
    x_t = x_t.detach().clone()
    p_indices, l_indices = torch.nonzero(cmd == 5, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 0:5] = 0.
        x_t[p_indices, l_indices, -3:] = 0.
        x_t[p_indices, l_indices, -5] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 4, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 0, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 2, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:4] = 0.
        x_t[p_indices, l_indices, 5:] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 1, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 4:] = 0.
        x_t[p_indices, l_indices, 3] = 0.
        x_t[p_indices, l_indices, 2] /= 180 / np.pi
    x_t = x_t[None]
    # print(x_t[0, p_indices, l_indices, 2])
    model.normalize(x_t, args_mask)
    # print(x_t[0, p_indices, l_indices, 2])
    return rearrange(x_t, 'B P L N -> B N P L')


def step_callback(
        gmflow,
        x_t,
        t,
        cmd,
        cmd_mask,
        args_mask,
        gt_arc_arg,
        model: GMFlowModel,
        save_intermediate=False,
        output_subdir: Path = None,
        part_names=None,
        base_args: torch.Tensor = None,
        rgb_gt: np.ndarray = None,
        sym_type: str = None,
    ):
    """Wrapper around symmetry_opt that optionally saves intermediate results per timestep.

    Saves to output_subdir/intermediate/step_XXXX/{intermediate.h5, intermediate.obj, intermediate.png}.
    """
    # Run the optimization step first
    x_t_new = symmetry_opt(gmflow, x_t, t, cmd, cmd_mask, args_mask, gt_arc_arg, model, sym_type)

    if not save_intermediate or output_subdir is None:
        return x_t_new

    try:
        step_id = int(t.item()) if torch.is_tensor(t) else int(t)
    except Exception:
        step_id = 0

    try:
        inter_root = output_subdir / 'intermediate'
        step_dir = inter_root / f'step_{step_id:04d}'
        step_dir.mkdir(parents=True, exist_ok=True)

        # Prepare current denormalized args for conversion to CAD vectors
        x_denorm = rearrange(x_t_new.clone(), 'B N P L -> B P L N')
        model.denormalize(x_denorm, args_mask)

        # Merge with base args (keep non-predicted values from base)
        if base_args is not None:
            args_current = base_args.clone()
            mask = args_mask[0] > 0
            args_current[mask] = x_denorm[0][mask]
        else:
            args_current = x_denorm[0]

        # Build CAD vectors
        cad_vec = []
        part_cad_vec_out = {}
        # import pdb; pdb.set_trace()
        _, P, L = cmd.shape
        for p in range(P):
            if cmd_mask[0][p][0] > 0:
                part_vec = []
                for l in range(L):
                    if cmd_mask[0][p][l] > 0:
                        arg = args_current[p][l].detach().cpu().numpy()
                        cmd_type = int(cmd[0][p][l].item())
                        if cmd_type == 1:
                            arg[2] /= np.pi / 180.
                            arg[3] = 1.0
                        cad_vec.append(np.concatenate([np.array([cmd_type]), arg]))
                        part_vec.append(cad_vec[-1])
                if part_names is not None and p < len(part_names):
                    part_cad_vec_out[part_names[p]] = np.array(part_vec)

        cad_vec_np = np.array(cad_vec)

        # Save h5
        with h5py.File(str(step_dir / 'intermediate.h5'), 'w') as h5_out:
            h5_out.create_dataset('vec', data=cad_vec_np)
            for k in part_cad_vec_out.keys():
                h5_out.create_dataset(k, data=part_cad_vec_out[k], dtype=float)

        # Save obj and png
        full_shape = vec2CADsolid(cad_vec_np, is_numerical=False)
        obj_path = step_dir / 'intermediate.obj'
        cad_to_obj(full_shape, str(obj_path))
        if rgb_gt is not None:
            rgb_pred = render_nvdiffrast_rgb(str(obj_path))
            rgb = np.concatenate([rgb_gt, rgb_pred], axis=1)
            cv2.imwrite(str(step_dir / 'intermediate.png'), rgb[..., ::-1])

    except Exception as e:
        # Never break sampling due to intermediate saving errors
        print(f"Error saving intermediate: {e}")
        import traceback
        print(traceback.format_exc())

    return x_t_new

def get_clip_features(clip_model, texts, device):
    """Get CLIP text features for given texts."""
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def compute_metrics_only(args, category: str):
    """Evaluate isolated-object outputs and save the same report schema as scenes."""
    import csv
    import zlib
    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f'No reconstructed outputs in {out_dir}; generate Stage 2 predictions first')
    ids_path = Path(args.data_root)/'splits'/f'{category}_test_ids.txt'
    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f'No test IDs in {ids_path}')
    rows=[]
    for data_id in tqdm(ids,desc=f'Original {category} metrics'):
        row=dict(sample_id=data_id,category=category,occ=0,cd=None,error=None,
                 partseg_accuracy=None,partseg_instance_avg_iou=None)
        try:
            mesh=trimesh.load(str(out_dir/data_id/'final.obj'),force='mesh')
            if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                raise ValueError('Invalid reconstructed mesh')
            row['occ']=1
            gt=trimesh.load(str(Path(args.data_root)/'raw_annotated'/category/data_id/'raw.obj'),force='mesh')
            cd=float(compute_cd(gt,mesh,normalize=True,seed=(args.seed+zlib.crc32(data_id.encode()))%(2**32)))
            if not np.isfinite(cd):
                raise ValueError('Non-finite CD')
            row['cd']=cd
        except Exception as error:
            row['error']=str(error)
        rows.append(row)
    segmentation={}
    if args.eval_partseg:
        records=[dict(sample_id=r['sample_id'],reference_id=r['sample_id']) for r in rows if r['occ']]
        try:
            metric=evaluate_partseg(str(out_dir),category.capitalize(),checkpoint_dir=args.partseg_ckpt_dir,
                                    data_root=args.data_root,batch_size=1,num_points=2048,num_votes=3,
                                    device=args.partseg_device or ('cuda' if torch.cuda.is_available() else 'cpu'),
                                    records=records,strict=True,seed=args.seed)
            indexed={r['sample_id']:r for r in rows}
            for item in metric.pop('per_sample',[]):
                indexed[item['sample_id']].update(partseg_accuracy=item['accuracy'],partseg_instance_avg_iou=item['instance_avg_iou'])
            segmentation[category]=dict(metric,status='success')
        except Exception as error:
            segmentation[category]=dict(status='unavailable',error=str(error),num_requested=len(records))
    overall=summarize_scene_metrics(rows)
    report=dict(dataset='original',category=category,split='test',output_dir=str(out_dir.resolve()),
                overall=overall,by_category={category:overall},part_segmentation=segmentation,
                partseg_requested=args.eval_partseg,
                definitions=dict(mean_occ='Successful mesh fraction',mean_cd='Original independently normalized per-object CD; not scene CD'))
    (out_dir/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out_dir/'metrics_per_object.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with (out_dir/'metrics_per_object.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Original {category} evaluation:',json.dumps(overall,indent=2))
    print(f'Metrics saved to {out_dir/"metrics.json"}')
    return report


def scene_records(args):
    records = load_records(args.dataset_root, args.split, args.object_count)
    category = getattr(args, 'category', None)
    if category and category != 'scene':
        records = [r for r in records if r['category'] == category]
    if args.max_samples:
        records = records[:args.max_samples]
    if not records:
        raise ValueError('No scene records match the requested category/count/split')
    return records


def summarize_scene_metrics(rows):
    """Keep generation failures in OCC; CD and segmentation use valid measurements."""
    cds = [r['cd'] for r in rows if r.get('cd') is not None]
    segmented = [r for r in rows if r.get('partseg_accuracy') is not None]
    return {
        'total_samples': len(rows),
        'successful_generations': sum(r['occ'] for r in rows),
        'mean_occ': float(np.mean([r['occ'] for r in rows])) if rows else None,
        'valid_cd_count': len(cds),
        'mean_cd': float(np.mean(cds)) if cds else None,
        'partseg_sample_count': len(segmented),
        'partseg_accuracy': float(np.mean([r['partseg_accuracy'] for r in segmented])) if segmented else None,
        'partseg_instance_avg_iou': float(np.mean([r['partseg_instance_avg_iou'] for r in segmented])) if segmented else None,
    }


def compute_scene_metrics(args):
    """Run the original geometry and PointNet metrics using scene sample identities."""
    import csv
    import zlib
    out_dir = args.output_dir or Path(args.data_root) / 'output/trassembler_h5/scene' / group_name(args.object_count) / args.split
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)
    records = scene_records(args)
    if getattr(args, 'original_eval', False):
        required = ('text_emb_retrieval', 'symmetry_refinement', 'use_gt_arc_args')
        mismatched = []
        for record in records:
            path = out_dir / record['sample_id'] / 'metadata.json'
            metadata = json.loads(path.read_text()) if path.is_file() else {}
            options = metadata.get('inference_options', {})
            if any(not options.get(key) for key in required):
                mismatched.append(record['sample_id'])
        if mismatched:
            raise ValueError(f"Original-evaluation settings were not used for {len(mismatched)} predictions; rerun inference with --original_eval first (examples: {mismatched[:3]})")
    rows = []
    valid_records = []
    rng_state = np.random.get_state()
    try:
        for record in tqdm(records, desc='Scene geometry metrics'):
            row = {key: record[key] for key in ('sample_id', 'scene_id', 'category', 'object_count')}
            row.update(reference_id=record.get('reference_id'), occ=0, cd=None, error=None,
                       partseg_accuracy=None, partseg_instance_avg_iou=None, model_checkpoint=None,
                       symmetry_refinement=None, reference_arc_assistance=None, text_emb_retrieval=None)
            folder = out_dir / record['sample_id']
            try:
                metadata_path = folder / 'metadata.json'
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text())
                    row['model_checkpoint'] = metadata.get('model_checkpoint')
                    options = metadata.get('inference_options', {})
                    row.update(symmetry_refinement=options.get('symmetry_refinement'),
                               reference_arc_assistance=options.get('use_gt_arc_args'),
                               text_emb_retrieval=options.get('text_emb_retrieval'))
                    if metadata.get('status') == 'failed':
                        raise ValueError(metadata.get('error', 'Reconstruction failed'))
                pred_path = folder / 'final.obj'
                predicted = trimesh.load_mesh(str(pred_path), force='mesh')
                if not len(predicted.faces) or not np.isfinite(predicted.vertices).all():
                    raise ValueError('Empty or non-finite reconstructed mesh')
                row['occ'] = 1
                valid_records.append(record)
                reference = args.dataset_root / Path(record['cad_target']).with_name('raw.obj')
                ground_truth = trimesh.load_mesh(str(reference), force='mesh')
                # Use the original normalized, bidirectional, unsquared CD unchanged.
                np.random.seed((args.seed + zlib.crc32(record['sample_id'].encode())) % (2 ** 32))
                cd = float(compute_cd(ground_truth, predicted, normalize=True, seed=(args.seed + zlib.crc32(record['sample_id'].encode())) % (2 ** 32)))
                if not np.isfinite(cd):
                    raise ValueError('Non-finite Chamfer distance')
                row['cd'] = cd
            except Exception as error:
                row['error'] = str(error)
            rows.append(row)
    finally:
        np.random.set_state(rng_state)
    segmentation = {}
    if args.eval_partseg:
        row_by_id = {r['sample_id']: r for r in rows}
        for category in sorted({r['category'] for r in records}):
            selected = [r for r in valid_records if r['category'] == category]
            if not selected:
                segmentation[category] = {'status': 'unavailable', 'error': 'No successful reconstructions', 'num_requested': 0}
                continue
            try:
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                metric = evaluate_partseg(
                    obj_dir=str(out_dir), category=category.capitalize(),
                    checkpoint_dir=args.partseg_ckpt_dir, data_root=args.data_root,
                    batch_size=1, num_points=2048, num_votes=3,
                    device=args.partseg_device or ('cuda' if torch.cuda.is_available() else 'cpu'),
                    records=selected, strict=True, verbose=True, seed=args.seed)
                for item in metric.pop('per_sample', []):
                    row_by_id[item['sample_id']].update(partseg_accuracy=item['accuracy'],
                                                       partseg_instance_avg_iou=item['instance_avg_iou'])
                segmentation[category] = dict(metric, status='success')
            except Exception as error:
                segmentation[category] = {'status': 'unavailable', 'error': str(error), 'num_requested': len(selected)}
        np.random.set_state(rng_state)
    report = {
        'dataset': 'scenes', 'split': args.split, 'object_count': args.object_count,
        'category': getattr(args, 'category', None), 'max_samples': args.max_samples,
        'output_dir': str(out_dir.resolve()), 'seed': args.seed,
        'partseg_requested': args.eval_partseg,
        'model_checkpoints': sorted({r['model_checkpoint'] for r in rows if r['model_checkpoint']}),
        'definitions': {
            'mean_occ': 'Successful reconstructed mesh fraction; NOT volumetric IoU',
            'mean_cd': 'Original compute_cd: 2048 surface points, independent centering/unit-radius normalization, sum of bidirectional mean Euclidean distances; lower is better',
            'partseg': 'Original category-specific PointNet2 accuracy/mIoU, original reference annotations and coordinate transform; missing annotations/checkpoints excluded and reported',
            'aggregation': 'Object-weighted means; partseg class_avg metrics are reported per category',
        },
        'overall': summarize_scene_metrics(rows),
        'by_object_count': {str(n): summarize_scene_metrics([r for r in rows if r['object_count'] == n]) for n in sorted({r['object_count'] for r in rows})},
        'by_category': {cat: summarize_scene_metrics([r for r in rows if r['category'] == cat]) for cat in sorted({r['category'] for r in rows})},
        'part_segmentation': segmentation,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out_dir / 'metrics_per_object.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    with (out_dir / 'metrics_per_object.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.eval_partseg and any(r.get('status') != 'success' for r in segmentation.values()):
        print('Part segmentation incomplete: see per-category availability/errors in metrics.json')
    print('Scene evaluation:', json.dumps(report['overall'], indent=2))
    print(f'Metrics saved to {out_dir / "metrics.json"}')
    return report


def load_structure(path, clip_model, data_id, image, name_embeddings=None):
    """Validate a scene prediction and use the shared inference sample builder."""
    with h5py.File(path) as stream:
        names = sorted(k for k in stream if k != 'vec' and 'bbox' not in k)
        if not names or any(not isinstance(stream[k], h5py.Dataset)
                            or stream[k].ndim != 2 or stream[k].shape[1] != 17
                            or not len(stream[k]) or not np.isfinite(stream[k][:]).all()
                            for k in names):
            raise ValueError(f'Invalid Stage 1 structure: {path}')
        sample = create_sample(stream, names, data_id, image, name_embeddings is not None, name_embeddings,
                               clip_model, group_by_base=True)
    return names, sample


def scene_refinement_inputs(record, args):
    """Resolve the same reference symmetry/arc inputs as original evaluation."""
    symmetry = 'no_symmetry'
    if args.symmetry_refinement:
        path = Path(args.sym_labels_dir) / f"{record['reference_id']}.txt"
        if not path.is_file():
            raise FileNotFoundError(f'Reference symmetry label missing: {path}')
        labels = path.read_text().splitlines()
        if not labels or labels[0].strip() not in ('1', '2', '3'):
            raise ValueError(f'Invalid reference symmetry label: {path}')
        symmetry = {1: 'rotational', 2: 'reflection', 3: 'no_symmetry'}[int(labels[0])]
    arc = None
    if args.use_gt_arc_args:
        with h5py.File(args.dataset_root / record['cad_target']) as stream:
            for name in sorted(stream):
                if name == 'vec' or 'bbox' in name or not isinstance(stream[name], h5py.Dataset):
                    continue
                vector = stream[name][:]
                arcs = vector[vector[:, 0] == 1]
                if len(arcs):
                    arc = arcs[0, 1:].copy()
                    break
    return symmetry, arc


def audit_scene_inputs(args):
    """Check original-style evaluation prerequisites without loading either model."""
    rows = []
    predictions = args.predictions_dir or Path(args.data_root) / 'output/llamaft_h5/scene' / group_name(args.object_count) / args.split
    output = args.output_dir or Path(args.data_root) / 'output/trassembler_h5/scene' / group_name(args.object_count) / args.split
    for record in scene_records(args):
        row = {key: record[key] for key in ('sample_id', 'reference_id', 'category', 'object_count')}
        row['errors'] = []
        structure = predictions / f"{record['sample_id']}.h5"
        if not structure.is_file():
            row['errors'].append(f'Missing structure prediction: {structure}')
        try:
            symmetry, arc = scene_refinement_inputs(record, args)
            row.update(symmetry_type=symmetry, reference_arc_available=arc is not None)
        except (OSError, ValueError, KeyError) as error:
            row['errors'].append(str(error))
        if args.text_emb_retrieval:
            dictionary = Path(args.data_root) / f"partnet2common_{record['category']}.json"
            if not dictionary.is_file():
                row['errors'].append(f'Missing text retrieval dictionary: {dictionary}')
        rows.append(row)
    report = dict(ready=not any(r['errors'] for r in rows), total=len(rows),
                  ready_samples=sum(not r['errors'] for r in rows),
                  options={key: getattr(args, key) for key in ('text_emb_retrieval', 'symmetry_refinement', 'use_gt_arc_args')},
                  samples=rows)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'evaluation_inputs.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Evaluation inputs: {report['ready_samples']}/{report['total']} ready; report: {path}")
    return report


def infer_scenes(args):
    from OCC.Extend.DataExchange import write_step_file

    report = audit_scene_inputs(args)
    if not report['ready']:
        raise ValueError('Scene evaluation prerequisites are missing; see evaluation_inputs.json. No model was loaded.')
    records = scene_records(args)
    # Validate reference-assisted inputs before allocating model memory.
    refinement_inputs = {r['sample_id']: scene_refinement_inputs(r, args) for r in records}
    if not torch.cuda.is_available():
        raise RuntimeError('Run scene inference inside a GPU allocation')
    if args.model_dir is None:
        raise ValueError('--model_dir is required for inference')
    model_dir = args.model_dir
    checkpoint = model_dir / 'checkpoints/last.ckpt'
    if not checkpoint.is_file():
        checkpoint = model_dir / 'last.ckpt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'No last.ckpt in {model_dir} or its checkpoints directory')
    config_path = model_dir / '.hydra' / 'config.yaml'
    if args.model_source == 'original':
        if args.category not in ('chair','table','storagefurniture'):
            raise ValueError('Original category checkpoints on scenes require --category chair, table, or storagefurniture')
        config = OmegaConf.load(config_path if config_path.is_file() else Path(__file__).with_name('config.yaml'))
        config.category = args.category
        config.dataset = 'original'
        config.task = 'geometry'
        stats_path = args.stats_path or model_dir/f'{args.category}_mean_scale_stats.json'
        if not stats_path.is_file():
            stats_path = Path(args.data_root)/f'{args.category}_mean_scale_stats.json'
        if args.predictions_dir is None or args.output_dir is None:
            raise ValueError('Specify separate --predictions_dir and --output_dir for original-model-on-scene experiments')
    else:
        if not config_path.is_file():
            raise FileNotFoundError(f'{config_path}: use a scene-trained experiment directory')
        config = OmegaConf.load(config_path)
        if config.get('task', 'geometry') != 'geometry':
            raise ValueError('Placement checkpoints require --task placement')
        if config.category != 'scene' or config.dataset != 'scenes':
            raise ValueError('Original category checkpoints on scene crops require --model_source original and --category')
        stats_path = args.stats_path or (model_dir/'mean_scale_stats.json' if (model_dir/'mean_scale_stats.json').is_file() else config.get('mean_scale_stats_path'))
        if not stats_path:
            stats_path = Path(args.data_root)/'trassembler_data/scene'/group_name(config.get('object_count'))/'mean_scale_stats.json'
    sampling_shift = args.sampler_shift if args.sampler_shift is not None else config.gm.shift
    model = GMFlowModel.load_from_checkpoint(
        checkpoint, args=config, embed_dim=config.network.embed_dim,
        num_heads=config.network.num_heads, dropout=config.network.dropout,
        bias=True, scaling_factor=1., args_range=np.array([-1., 1.]), shift=sampling_shift,
        mean_scale_stats_path=stats_path, pretrained_backbone=False,
        map_location='cpu').cuda().eval()
    clip_model, _ = clip.load('ViT-B/32', device='cuda', download_root=str(ROOT / 'data/.cache/clip'))
    name_databases = {}
    group = group_name(args.object_count)
    predictions = args.predictions_dir or Path(args.data_root) / 'output/llamaft_h5/scene' / group / args.split
    output = args.output_dir or Path(args.data_root) / 'output/trassembler_h5/scene' / group / args.split
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    options = {key: getattr(args, key) for key in ('text_emb_retrieval', 'save_intermediate', 'symmetry_refinement', 'use_gt_arc_args', 'no_preview')}
    options.update(model_source=args.model_source, sampler_shift=float(sampling_shift))
    prepared_manifest = Path(args.data_root) / 'trassembler_data/scene' / group_name(config.get('object_count')) / f'{args.split}.json'
    prepared_metadata = {}
    if prepared_manifest.is_file():
        prepared_metadata = {r['sample_id']: r for r in json.loads(prepared_manifest.read_text())}
    results = []
    for record in records:
        object_dir = output / record['sample_id']
        object_dir.mkdir(parents=True, exist_ok=True)
        result = dict(record, model_checkpoint=str(checkpoint.resolve()),
                      structure_prediction=str((predictions / (record['sample_id'] + '.h5')).resolve()),
                      output_dir=str(object_dir.resolve()), coordinate_system='normalized_object_local',
                      checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                      inference_options=options,
                      placement='not_applied; scene_annotations retained for later assembly')
        reference_scale = prepared_metadata.get(record['sample_id'], {}).get('source_to_normalized_scale')
        if reference_scale is not None:
            # Reference metadata for later annotated assembly, never used in sampling.
            result['reference_source_to_normalized_scale'] = reference_scale
        try:
            previous = {}
            metadata_path = object_dir / 'metadata.json'
            if metadata_path.is_file():
                try:
                    previous = json.loads(metadata_path.read_text())
                except (ValueError, OSError):
                    pass
            complete = (previous.get('status') in ('success', 'skipped')
                        and previous.get('model_checkpoint') == result['model_checkpoint']
                        and previous.get('checkpoint_mtime_ns') == result['checkpoint_mtime_ns']
                        and previous.get('inference_options') == options
                        and all((object_dir / f'final.{ext}').is_file() for ext in ('h5', 'obj', 'step')))
            if args.skip_processed and complete:
                result['status'] = 'skipped'
            else:
                name_embeddings = None
                if args.text_emb_retrieval:
                    category = record['category']
                    if category not in name_databases:
                        names_path = Path(args.data_root) / f'partnet2common_{category}.json'
                        vocabulary = sorted({name.split('/')[0] for name in json.loads(names_path.read_text()).values()})
                        name_databases[category] = get_clip_features(clip_model, vocabulary, 'cuda').cpu().numpy()
                    name_embeddings = name_databases[category]
                image = crop_tensor(args.dataset_root / record['image'])
                rgb_gt = np.moveaxis(np.clip((image + 1.) * 127.5, 0, 255).astype(np.uint8), 0, -1)
                names, sample = load_structure(
                    predictions / f"{record['sample_id']}.h5", clip_model,
                    record['sample_id'], image, name_embeddings)
                batch = collate_fn([sample])
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                sym_type, arc = refinement_inputs[record['sample_id']]
                gt_arc_arg = torch.tensor(arc, device='cuda') if arc is not None else None
                result['symmetry_type'] = sym_type
                result['reference_arc_available'] = arc is not None
                callback = None
                if args.symmetry_refinement or args.save_intermediate:
                    callback = partial(step_callback, cmd=batch['command'], cmd_mask=batch['command_mask'],
                                       args_mask=batch['args_mask'], gt_arc_arg=gt_arc_arg, model=model,
                                       save_intermediate=args.save_intermediate, output_subdir=object_dir,
                                       part_names=names, base_args=batch['args'][0].clone().detach(),
                                       rgb_gt=None if args.no_preview else rgb_gt, sym_type=sym_type)
                with torch.no_grad():
                    predicted = model.sample(batch, step_callback=callback)
                    model.denormalize(predicted, batch['args_mask'])
                parts = decode_vectors(batch, predicted)
                if len(parts) != len(names):
                    raise ValueError('Decoded part count differs from input structure')
                vector = np.concatenate(parts)
                if not np.isfinite(vector).all():
                    raise ValueError('Non-finite predicted geometry')
                with h5py.File(object_dir / 'final.h5', 'w') as stream:
                    stream.create_dataset('vec', data=vector)
                    for name, part in zip(names, parts):
                        stream.create_dataset(name, data=part)
                    stream.attrs['sample_id'] = record['sample_id']
                    stream.attrs['scene_id'] = str(record['scene_id'])
                    stream.attrs['coordinate_system'] = 'normalized_object_local'
                shape = vec2CADsolid(vector, is_numerical=False)
                cad_to_obj(shape, str(object_dir / 'final.obj'))
                write_step_file(shape, str(object_dir / 'final.step'))
                if not (object_dir / 'final.step').is_file():
                    raise RuntimeError('STEP export did not produce a file')
                result['status'] = 'success'
                if not args.no_preview:
                    try:
                        preview = render_nvdiffrast_rgb(str(object_dir / 'final.obj'))
                        cv2.imwrite(str(object_dir / 'final.png'), np.concatenate([rgb_gt, preview], axis=1)[..., ::-1])
                    except Exception as error:
                        result['preview_error'] = str(error)
                        print(f"Preview failed for {record['sample_id']}: {error}", flush=True)
        except Exception as error:
            result.update(status='failed', error=str(error))
        (object_dir / 'metadata.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        # Update after each object so interrupted runs retain their progress.
        (output / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
        print(record['sample_id'], result['status'], result.get('error', ''), flush=True)
    args.output_dir = output
    compute_scene_metrics(args)
    failures = sum(r['status'] == 'failed' for r in results)
    print(f'{len(results)} processed; {failures} failed; outputs: {output}')
    if failures:
        raise SystemExit(1)




def evaluate_placement_task(args):
    from SceneAssembler.placement import (SCHEMA, load_samples, group_name as placement_group,
                                          vector_to_matrix, evaluate, read_predictions)
    output = Path(args.output_dir) if args.output_dir else Path(args.data_root)/'output/trassembler_placement/scene'/placement_group(args.object_count)/args.split
    targets = None
    if not args.metrics_only and output.exists() and any(output.glob('*/final.h5')):
        raise ValueError('Keep placement outputs separate from CAD geometry outputs')
    if args.metrics_only:
        path = args.placement_predictions or output/'predictions.jsonl'
        targets = load_samples(args.dataset_root,args.split,args.object_count)
        if args.max_samples:
            targets = targets[:args.max_samples]
        return evaluate(read_predictions(path),targets,output)
    if not torch.cuda.is_available():
        raise RuntimeError('Run placement prediction inside a GPU allocation; metrics_only does not require a GPU')
    from dataset import ScenePlacementDataset
    from model import PlacementModel
    from torch.utils.data import DataLoader
    directory = Path(args.model_dir)
    if not (directory/'placement_config.json').is_file():
        raise ValueError('Use a placement-trained run directory, not a geometry checkpoint')
    checkpoint = directory/'checkpoints/last.ckpt'
    model = PlacementModel.load_from_checkpoint(checkpoint,map_location='cpu',pretrained_backbone=False).cuda().eval()
    dataset = ScenePlacementDataset(args.dataset_root,args.split,args.object_count,include_targets=False)
    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
    output.mkdir(parents=True,exist_ok=True)
    predictions=[]
    torch.manual_seed(args.seed)
    for sample,batch in zip(dataset.samples,DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)):
        item={k:sample[k] for k in ('sample_id','scene_id','category','object_count')}
        item.update(schema=SCHEMA,checkpoint=str(checkpoint.resolve()),coordinate_system='normalized_object_local',
                    target_coordinate_system='scene',supplied_boxes=True,supplied_camera=True)
        try:
            batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with torch.no_grad():
                vector=model.predict_vectors(batch)[0].cpu().numpy()
            item.update(status='success',normalized_to_scene=vector_to_matrix(vector).tolist())
        except Exception as error:
            item.update(status='failed',error=str(error))
        predictions.append(item)
        (output/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
        print(sample['sample_id'],item['status'],flush=True)
    targets=load_samples(args.dataset_root,args.split,args.object_count)
    selected={s['sample_id'] for s in dataset.samples}
    report=evaluate(predictions,[s for s in targets if s['sample_id'] in selected],output)
    if any(p['status']!='success' for p in predictions):
        raise SystemExit(1)
    return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate TrAssembler model.")
    parser.add_argument("--task", choices=["geometry", "placement"], default="geometry")
    parser.add_argument("--placement_predictions", type=Path, help="Saved placement predictions for metrics_only")
    parser.add_argument('--model_source', choices=['scenes','original'], default='scenes', help='Checkpoint family when evaluating scene crops')
    parser.add_argument('--direct_sampling', action='store_true', help='Disable reference-assisted refinement for controlled original/scene comparisons')
    parser.add_argument('--sampler_shift', type=float, default=None, help='Explicit common sampling shift for scene-domain model comparisons')
    parser.add_argument("--model_dir", type=str, default='data/ckpts/trassembler/chair', help="Model directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults depend on dataset).")
    parser.add_argument("--text_emb_retrieval", action='store_true', help="Use text embedding retrieval.")
    parser.add_argument("--data_root", type=str, default='data', help="Data root directory.")
    parser.add_argument("--skip_processed", action='store_true', help="Skip processed objects.")
    parser.add_argument("--sym_labels_dir", type=str, default='data/sym_labels', help="Symmetry label directory.")
    parser.add_argument("--save_intermediate", action='store_true', help="Save intermediate step outputs (h5, obj, png) per timestep under out_dir/data_id/intermediate.")
    parser.add_argument("--eval_partseg", action='store_true', help="Evaluate part segmentation accuracy and mIoU.")
    parser.add_argument("--partseg_ckpt_dir", type=str, default='data/ckpts/partseg', help="Part segmentation checkpoint directory.")
    parser.add_argument("--metrics_only", action='store_true', help="Skip inference, only compute metrics from existing outputs.")
    parser.add_argument("--category", type=str, default=None, help="Category for metrics-only mode (chair, table, storagefurniture). If not set, inferred from model_dir config.")
    
    parser.add_argument('--dataset', choices=['original', 'scenes'], default='original')
    parser.add_argument('--dataset_root', type=Path, default=None, help='Scene dataset root; defaults to data_root/scene_dataset_v2')
    parser.add_argument('--object_count', type=int, choices=[1, 2, 3, 4], help='Scene size; omit to pool all sizes')
    parser.add_argument('--predictions_dir', type=Path, help='Override scene LlamaFT prediction directory')
    parser.add_argument('--stats_path', type=Path, help='Override scene normalization statistics')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_preview', action='store_true', help='Skip scene rendering; CAD and geometry metrics are still saved')
    parser.add_argument('--partseg_device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--check_inputs', action='store_true', help='Audit scene evaluation prerequisites without loading a model')
    parser.add_argument('--original_eval', action='store_true', help='Scene preset: original README text retrieval plus reference symmetry and GT arc assistance')
    parser.add_argument('--symmetry_refinement', action='store_true', help='Apply original refinement to scene objects using reference symmetry labels')
    parser.add_argument('--use_gt_arc_args', action='store_true', help='Use reference arc direction during scene refinement (ground-truth-assisted experiment)')

    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max_samples must be positive')
    if args.task == 'placement':
        args.dataset = 'scenes'
        if args.eval_partseg or args.save_intermediate or args.check_inputs or args.skip_processed:
            parser.error('CAD segmentation/intermediate/audit/skip flags do not apply to placement; its evaluator regenerates the placement manifest')
        if args.category not in (None, 'scene'):
            parser.error('Placement pools furniture categories; use the per-category metric breakdown')
        args.dataset_root = args.dataset_root or Path(args.data_root)/'scene_dataset_v2'
        if args.original_eval or args.symmetry_refinement or args.use_gt_arc_args or args.text_emb_retrieval:
            parser.error('CAD refinement/retrieval flags do not apply to placement evaluation')
        evaluate_placement_task(args)
        return
    if args.sampler_shift is not None and args.sampler_shift <= 0:
        parser.error('--sampler_shift must be positive')
    if args.direct_sampling and (args.original_eval or args.symmetry_refinement or args.use_gt_arc_args):
        parser.error('--direct_sampling cannot be combined with reference-assisted refinement')
    if args.category == 'scene':
        args.dataset = 'scenes'
    if args.dataset == 'scenes':
        if args.original_eval:
            args.text_emb_retrieval = True
            args.symmetry_refinement = True
            args.use_gt_arc_args = True
        if args.category not in (None, 'scene', 'chair', 'table', 'storagefurniture'):
            parser.error('Scene category must be scene, chair, table, or storagefurniture')
        args.dataset_root = args.dataset_root or Path(args.data_root) / 'scene_dataset_v2'
        if args.use_gt_arc_args and not args.symmetry_refinement and not args.metrics_only:
            parser.error('--use_gt_arc_args requires --symmetry_refinement')
        args.model_dir = Path(args.model_dir)
        args.output_dir = Path(args.output_dir) if args.output_dir else None
        if args.check_inputs:
            report = audit_scene_inputs(args)
            if not report['ready']:
                raise SystemExit(2)
        elif args.metrics_only:
            compute_scene_metrics(args)
        else:
            infer_scenes(args)
        return
    if args.original_eval:
        parser.error('--original_eval requires --dataset scenes')
    if args.object_count is not None:
        parser.error('--object_count requires --dataset scenes')
    args.output_dir = args.output_dir or 'data/output/trassembler_h5'

    
    # Metrics-only mode: skip inference, just compute metrics from existing outputs
    if args.metrics_only:
        # Determine category
        if args.category:
            category = args.category
        else:
            # Try to infer from output_dir path
            out_dir_parts = Path(args.output_dir).parts
            if out_dir_parts:
                category = out_dir_parts[-1]
            else:
                print("Error: --category must be specified in metrics-only mode if not inferable from output_dir")
                return
        
        compute_metrics_only(args, category)
        return
    
    # Normal inference mode
    save_intermediate = args.save_intermediate
    skip_processed = args.skip_processed
    sym_labels_dir = args.sym_labels_dir
    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / 'checkpoints' / 'last.ckpt'
    if not ckpt_path.is_file():
        ckpt_path = model_dir / 'last.ckpt'
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'No last.ckpt found in {model_dir} or its checkpoints directory')
    config_path = model_dir / '.hydra' / 'config.yaml'
    config = OmegaConf.load(config_path if config_path.is_file() else Path(__file__).with_name('config.yaml'))
    if args.category:
        config.category = args.category
    config.dataset = 'original'
    stats_path = args.stats_path or model_dir/f'{config.category}_mean_scale_stats.json'
    if not stats_path.is_file():
        stats_path = Path(args.data_root)/f'{config.category}_mean_scale_stats.json'
    h5_root = args.predictions_dir or Path(args.data_root)/'output/llamaft_h5'/config.category
    model = GMFlowModel.load_from_checkpoint(ckpt_path,
                                                   args=config, embed_dim=config.network.embed_dim, num_heads=config.network.num_heads, dropout=config.network.dropout,
                                  bias=True, scaling_factor=1., args_range=np.array([-1., 1.]),
                                  mean_scale_stats_path=stats_path, pretrained_backbone=False,
                                  shift=args.sampler_shift if args.sampler_shift is not None else 1.,
                                  map_location='cpu').cuda().eval()

    image_transform = Compose(
        [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    
    args.output_dir = os.path.join(args.output_dir, config.category)


    clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
    name_db = list(set(json.load(open(f'data/partnet2common_{config.category}.json')).values()))
    name_db = [name[:name.find('/')] if '/' in name else name for name in name_db]
    name_embeddings_db = clip_model.encode_text(clip.tokenize(name_db).to('cuda')).cpu().numpy()
    name_embeddings_db /= np.linalg.norm(name_embeddings_db, axis=1, keepdims=True)
    print(f"Name embeddings shape: {name_embeddings_db.shape}")
    print(f"Name database: {name_db}")

    np.random.seed(0)
    do_retrieval = args.text_emb_retrieval
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    test_ids = open(f'data/splits/{config.category}_test_ids.txt').read().splitlines()
    test_ids = [int(i) for i in test_ids]

    gt_data_root = Path(f'data/raw_annotated/{config.category}')

    occs = []
    cds = []
    cnt = 0
    
    sym_label2type = {1: 'rotational', 2: 'reflection', 3: 'no_symmetry'}
    for data_id in tqdm(test_ids):
        sym_type = 'no_symmetry'
        if not args.direct_sampling:
            sym_label_path = f'{sym_labels_dir}/{data_id}.txt'
            if not os.path.exists(sym_label_path):
                print(f'{sym_label_path} not exists')
                occs.append(0.)
                continue
            sym_label = open(sym_label_path).read().splitlines()[0]
            sym_type = sym_label2type[int(sym_label)]
        # Prepare per-object output directory
        obj_out_dir = out_dir / f'{data_id}'
        obj_out_dir.mkdir(exist_ok=True, parents=True)

        # if already processed, skip
        final_obj_path = obj_out_dir / 'final.obj'
        if os.path.exists(str(final_obj_path)) and skip_processed:
            obj_path = str(final_obj_path)
            occs.append(1.)

            gt_mesh_path = f'{gt_data_root}/{data_id}/raw.obj'
            if os.path.exists(gt_mesh_path):
                gt_mesh = trimesh.load_mesh(gt_mesh_path)
                pred_mesh = trimesh.load_mesh(obj_path)
                cd = compute_cd(gt_mesh, pred_mesh, normalize=True)
                cds.append(cd)
            continue
        h5_path = h5_root / f'{data_id}.h5'
        if not h5_path.exists():
            print(f'{h5_path} not exists')
            occs.append(0.)
            continue
        h5 = h5py.File(h5_path, 'r')
        part_names = list(sorted([n for n in h5 if n != 'vec' and '_bbox' not in n]))  # this has no effect, h5 will treat / as hierarchy

        if not args.direct_sampling:
            gt_h5_path = f'data/raw_annotated/{config.category}/{data_id}/cad.h5'
            gt_h5 = h5py.File(gt_h5_path, 'r')
            gt_part_names = list(sorted([n for n in gt_h5 if n != 'vec' and '_bbox' not in n]))  # this has no effect, h5 will treat / as hierarchy
        cnt += 1


        image_path = f'data/blender_renderings/{data_id}.png'
        image = cv2.imread(image_path)[..., ::-1].copy()
        image = image_transform(image)
        rgb_gt = cv2.imread(image_path)[..., ::-1].copy()

        sample = create_sample(h5, part_names, data_id, image, do_retrieval, name_embeddings_db, clip_model)
        if not args.direct_sampling:
            gt_sample = create_sample(gt_h5, gt_part_names, data_id, image, do_retrieval, name_embeddings_db, clip_model)

        batch = collate_fn([sample])
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
                
        gt_arc_arg = None
        if not args.direct_sampling:
            gt_batch = collate_fn([gt_sample])
            for k in gt_batch:
                if isinstance(gt_batch[k], torch.Tensor):
                    gt_batch[k] = gt_batch[k].cuda()
            gt_args = gt_batch['args'][0].clone().detach()
            gt_cmd = gt_batch['command'][0].clone().detach()
            arc_cmd = gt_cmd == 1
            if arc_cmd.any():
                gt_arc_arg = gt_args[arc_cmd][0]
            else:
                gt_arc_arg = None

        try:
            step_cb = partial(
                step_callback,
                cmd=batch['command'],
                cmd_mask=batch['command_mask'],
                args_mask=batch['args_mask'],
                gt_arc_arg=gt_arc_arg,
                model=model,
                save_intermediate=save_intermediate,
                output_subdir=obj_out_dir,
                part_names=part_names,
                base_args=batch['args'][0].clone().detach(),
                rgb_gt=rgb_gt,
                sym_type=sym_type,
            )
            pred_args = model.sample(batch, step_callback=step_cb)
        except Exception as e:
            print(f"Error during sampling: {e}")
            occs.append(0.)
            continue
        # Denormalize the predicted args
        model.denormalize(pred_args, batch['args_mask'])

        # Assuming batch size is always 1 for evaluation
        i = 0 # Since batch size is 1
        data_id = batch['data_id'][i] # Use the actual data_id from the batch
        # rgb_gt already loaded above

        cad_vec = []
        cmd = batch['command'][i]  # [P, L]
        cmd_mask = batch['command_mask'][i]  # [P, L]
        arg_mask = batch['args_mask'][i]  # [P, L, N]
        args = batch['args'][i]  # [P, L, N]
        args[arg_mask > 0] = pred_args[i][arg_mask > 0]  # overwrite the args with predicted args

        part_cad_vec_out = {} # Renamed to avoid conflict
        for p in range(cmd_mask.shape[0]):
            if cmd_mask[p][0] > 0:  # this part is valid
                part_vec = []
                for l in range(cmd_mask.shape[1]):
                    if cmd_mask[p][l] > 0:
                        arg = args[p][l].cpu().numpy()
                        cmd_type = cmd[p][l].item()
                        if cmd_type == 1: # ARC
                            # Ensure angle is within valid range if needed, though denormalization should handle it
                            arg[2] /= np.pi / 180. # Convert back to degrees if needed by vec2CADsolid
                            arg[3] = 1.0
                        cad_vec.append(np.concatenate([np.array([cmd_type]), arg]))
                        part_vec.append(cad_vec[-1])

                # Use original part names from the loaded h5 file
                part_cad_vec_out[part_names[p]] = np.array(part_vec)
        cad_vec = np.array(cad_vec)
        try:
            full_shape = vec2CADsolid(cad_vec, is_numerical=False)
            # save obj
            obj_path = str(final_obj_path)
            cad_to_obj(full_shape, obj_path)
            rgb_pred = render_nvdiffrast_rgb(obj_path)
            # vis.image(np.moveaxis(rgb, -1, 0), win='val', opts={'title': 'val'})
            occs.append(1.)

            gt_mesh_path = f'{gt_data_root}/{data_id}/raw.obj'
            if os.path.exists(gt_mesh_path):
                gt_mesh = trimesh.load_mesh(gt_mesh_path)
                pred_mesh = trimesh.load_mesh(obj_path)
                cd = compute_cd(gt_mesh, pred_mesh, normalize=True)
                cds.append(cd)
            
        except Exception as e:
            print(f"Error processing {data_id}: {e}")
            rgb_pred = np.zeros((400, 400, 3), dtype=np.uint8)
            occs.append(0.)
            continue
        rgb = np.concatenate([rgb_gt, rgb_pred], axis=1)

        cv2.imwrite(str(obj_out_dir / 'final.png'), rgb[..., ::-1])

        # save h5 and the obj
        with h5py.File(str(obj_out_dir / 'final.h5'), 'w') as h5_out: # Renamed handle
            h5_out.create_dataset('vec', data=cad_vec)
            for part_name in part_cad_vec_out.keys():
                h5_out.create_dataset(part_name, data=part_cad_vec_out[part_name], dtype=float)

        metadata = dict(sample_id=str(data_id), category=config.category, dataset='original',
                        model_checkpoint=str(ckpt_path.resolve()), checkpoint_mtime_ns=ckpt_path.stat().st_mtime_ns,
                        status='success', coordinate_system='normalized_object_local',
                        inference_options=dict(text_emb_retrieval=args.text_emb_retrieval,
                                               symmetry_refinement=not args.direct_sampling, use_gt_arc_args=not args.direct_sampling,
                                               sampler_shift=float(model.shift)))
        (obj_out_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

    # Calculate metrics ignoring NaNs
    valid_cds = [c for c in cds if not np.isnan(c)]
    mean_occ = np.mean(occs) if occs else 0.0
    mean_cd = np.mean(valid_cds) if valid_cds else float('nan')

    print(f'\n{"="*60}')
    print(f'Geometry Evaluation Results ({config.category})')
    print(f'{"="*60}')
    print(f'Total processed: {len(test_ids)}')
    print(f'Successful generations (OCC > 0): {sum(occs)}')
    print(f'Valid CD computations: {len(valid_cds)}')
    print(f'Mean OCC: {mean_occ:.4f}')
    print(f'Mean CD (valid only): {mean_cd:.4f}')

    # Part Segmentation Evaluation
    if args.eval_partseg:
        print(f'\n{"="*60}')
        print(f'Part Segmentation Evaluation ({config.category})')
        print(f'{"="*60}')
        
        partseg_metrics = evaluate_partseg(
            obj_dir=str(out_dir),
            category=config.category.capitalize() if config.category != 'storagefurniture' else 'Storagefurniture',
            checkpoint_dir=args.partseg_ckpt_dir,
            data_root=args.data_root,
            batch_size=32,
            num_points=2048,
            num_votes=3,
            verbose=True,
        )
        
        print(f'\nSummary:')
        print(f'  Part Segmentation Accuracy: {partseg_metrics["accuracy"]:.4f}')
        print(f'  Part mIoU (Instance Avg): {partseg_metrics["instance_avg_iou"]:.4f}')


if __name__ == "__main__":
    main()
