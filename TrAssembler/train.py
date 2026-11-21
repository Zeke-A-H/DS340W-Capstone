#!/usr/bin/env python3
"""
Training script for TrAssembler model.

This script trains the TrAssembler model that predicts continuous parameters
given discrete CAD command structures.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model import GMFlowModel, CADDataModule
import numpy as np


def setup_logger(args, experiment_dir):
    """Setup logger based on configuration."""
    # Create logging directory under experiment directory
    log_dir = Path(experiment_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.wandb:
        logger = WandbLogger(
            project='Img2CAD-TrAssembler', 
            name=f'{args.category}_experiment',
            save_dir=str(log_dir)
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(log_dir), 
            name=f'TrAssembler_{args.category}'
        )
    return logger


def setup_callbacks(args, experiment_dir):
    """Setup training callbacks."""
    # Create checkpoint directory under experiment directory
    checkpoint_dir = Path(experiment_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch}-{step}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,
    )
    return [checkpoint_callback]


@hydra.main(config_path='.', config_name='config', version_base=None)
def train(args: DictConfig):
    """Main training function."""
    # Get Hydra output directory and derive experiment directory
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    
    # Keep checkpoints alongside the Hydra config for this individual run.
    experiment_dir = hydra_output_dir
    
    placement_marker = Path(experiment_dir)/'placement_config.json'
    if args.task == 'geometry' and placement_marker.is_file():
        raise ValueError('This output folder belongs to placement; keep geometry checkpoints separate')
    if args.task == 'placement' and not placement_marker.is_file() and any(
            path.is_file() for path in (Path(experiment_dir)/'checkpoints/last.ckpt', Path(experiment_dir)/'last.ckpt')):
        raise ValueError('Refusing to overwrite untagged geometry checkpoints with a placement model')

    print(f"Hydra output directory: {hydra_output_dir}")
    print(f"Experiment directory: {experiment_dir}")
    
    # Set random seeds for reproducibility
    pl.seed_everything(args.seed)
    
    # Setup logger
    logger = setup_logger(args, experiment_dir)
    
    # Setup callbacks
    callbacks = setup_callbacks(args, experiment_dir)
    
    if args.task == 'placement':
        if args.dataset != 'scenes' or args.category != 'scene':
            raise ValueError('Placement training requires dataset=scenes category=scene')
        if args.get('ckpt'):
            raise ValueError('Geometry checkpoints cannot initialize the new placement head; use resume_from_checkpoint for a placement checkpoint')
        from dataset import ScenePlacementDataset
        from model import PlacementModel
        from SceneAssembler.placement import fit_statistics
        from torch.utils.data import DataLoader
        import json
        train_dataset = ScenePlacementDataset(args.dataset_root,'train',args.object_count)
        val_dataset = ScenePlacementDataset(args.dataset_root,'val',args.object_count)
        statistics = fit_statistics(train_dataset.samples)
        resume = args.get('resume_from_checkpoint')
        if resume:
            old = Path(resume).parent.parent/'placement_config.json'
            if not old.is_file():
                raise ValueError('Expected a placement checkpoint with placement_config.json')
            metadata = json.loads(old.read_text())
            if metadata['object_count'] != args.object_count or metadata['dataset_root'] != str(Path(args.dataset_root).resolve()):
                raise ValueError('Placement resume data selection differs from the saved run')
            if metadata['statistics'].get('data_signature') != statistics['data_signature']:
                raise ValueError('Placement training data changed since the checkpoint')
            statistics = metadata['statistics']
        model = PlacementModel(statistics,lr=float(args.lr),hidden_dim=args.placement_hidden_dim,pretrained_backbone=not bool(resume))
        metadata = dict(task='placement',object_count=args.object_count,dataset_root=str(Path(args.dataset_root).resolve()),
                        statistics=statistics,supplied_boxes=True,supplied_camera=True,
                        transform='normalized_object_local_to_scene')
        (Path(experiment_dir)/'placement_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
        trainer = pl.Trainer(max_epochs=args.max_epochs,accelerator='cuda' if args.use_gpu else 'cpu',
                             callbacks=callbacks,logger=logger,fast_dev_run=args.fast_dev_run,
                             gradient_clip_val=args.gradient_clip_val,precision=args.precision,
                             limit_val_batches=args.limit_val_batches)
        trainer.fit(model,train_dataloaders=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0),
                    val_dataloaders=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0),
                    ckpt_path=resume)
        print('Placement training completed!')
        return

    if args.dataset == 'scenes':
        from dataset import group_name
        if args.object_count not in (None, 1, 2, 3, 4):
            raise ValueError('object_count must be null, 1, 2, 3, or 4')
        if args.get('ckpt') and not args.mean_scale_stats_path:
            raise ValueError('When fine-tuning a checkpoint, set mean_scale_stats_path to its original statistics')
        if args.category != 'scene':
            raise ValueError('Scene training requires category=scene')
        data_dir = f'data/trassembler_data/scene/{group_name(args.object_count)}'
        stats_path = args.mean_scale_stats_path or f'{data_dir}/mean_scale_stats.json'
    else:
        data_dir = f'data/trassembler_data/{args.category}_pkl'
        stats_path = args.mean_scale_stats_path

    resume_checkpoint = args.get('resume_from_checkpoint')
    if resume_checkpoint:
        if args.get('ckpt'):
            raise ValueError('Choose resume_from_checkpoint for full resume or ckpt for weights-only fine-tuning')
        if not Path(resume_checkpoint).is_file():
            raise FileNotFoundError(resume_checkpoint)
        if args.dataset == 'scenes' and not args.mean_scale_stats_path:
            raise ValueError('Full scene resume requires mean_scale_stats_path from the original run')

    if stats_path is not None:
        import shutil
        saved_stats = Path(experiment_dir) / 'mean_scale_stats.json'
        if Path(stats_path).resolve() != saved_stats.resolve():
            shutil.copyfile(stats_path, saved_stats)
        stats_path = str(saved_stats)

    # Setup data module
    data_module = CADDataModule(
        data_dir=data_dir,
        cat=args.category, dataset=args.dataset,
        batch_size=args.batch_size, 
    )
    
    # Load model from checkpoint if specified
    if hasattr(args, 'ckpt') and args.ckpt is not None:
        print(f"Loading model from {args.ckpt}")
        model = GMFlowModel.load_from_checkpoint(
            args.ckpt,
            mean_scale_stats_path=stats_path,
            args=args,
            embed_dim=args.network.embed_dim,
            num_heads=args.network.num_heads,
            dropout=args.network.dropout,
            bias=True,
            scaling_factor=1.,
            args_range=np.array([-1., 1.]),
            shift=args.gm.shift
        )
    else:
        if resume_checkpoint:
            print(f"Restoring full training state from {resume_checkpoint}")
        else:
            print("Training from scratch")
        model = GMFlowModel(
            mean_scale_stats_path=stats_path,
            args=args, 
            embed_dim=args.network.embed_dim,
            num_heads=args.network.num_heads, 
            dropout=args.network.dropout,
            bias=True, 
            scaling_factor=1., 
            args_range=np.array([-1., 1.]),
            shift=args.gm.shift
        )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        accelerator='cuda' if args.use_gpu else 'cpu',
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        precision=args.precision if hasattr(args, 'precision') else 32,
    )
    
    # Start training
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)
    
    print("Training completed!")


if __name__ == '__main__':
    train()
