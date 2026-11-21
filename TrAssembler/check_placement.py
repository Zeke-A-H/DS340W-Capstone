"""CPU smoke check of the new placement head without downloading/loading DINO weights."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model import PlacementModel
from SceneAssembler.placement import vector_to_matrix


class TinyBackbone(nn.Module):
    embed_dim=8
    def __init__(self):
        super().__init__()
        self.features=nn.Conv2d(3,8,1)
    def forward_features(self,image):
        return {'x_norm_patchtokens':self.features(image).flatten(2).transpose(1,2)}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    statistics={name:dict(mean=[0.]*dim,std=[1.]*dim) for name,dim in [('features',24),('translation',3),('log_scale',1)]}
    model=PlacementModel(statistics,hidden_dim=32,backbone=TinyBackbone())
    records=[]
    for i in range(4):
        records.append(dict(full_image=torch.rand(3,8,8),crop_image=torch.rand(3,8,8),features=torch.rand(24),
                            target=torch.tensor([i*.1,0.,0.,1.,0.,0.,0.,1.,0.,.1])))
    loader=DataLoader(records,batch_size=4)
    batch=next(iter(loader))
    frozen={k:v.clone() for k,v in model.backbone.state_dict().items()}
    optimizer=model.configure_optimizers()
    initial=float(model.loss(batch).detach())
    for _ in range(80):
        optimizer.zero_grad();loss=model.loss(batch);assert torch.isfinite(loss);loss.backward();optimizer.step()
    final=float(model.loss(batch).detach())
    assert final<initial
    assert all(torch.equal(frozen[k],v) for k,v in model.backbone.state_dict().items())
    with torch.no_grad():
        for vector in model.predict_vectors(batch).numpy():
            matrix=vector_to_matrix(vector)
            assert np.isfinite(matrix).all() and np.linalg.det(matrix[:3,:3])>0
    trainer=pl.Trainer(accelerator='cpu',fast_dev_run=True,logger=False,enable_checkpointing=False,enable_model_summary=False)
    trainer.fit(model,train_dataloaders=loader,val_dataloaders=loader)
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'placement.ckpt';trainer.save_checkpoint(path)
        restored=PlacementModel.load_from_checkpoint(path,backbone=TinyBackbone(),map_location='cpu')
        assert all(torch.equal(value,restored.state_dict()[key]) for key,value in model.state_dict().items())
    print(f'Placement head learning ({initial:.5f} -> {final:.5f}), frozen encoder, valid decoded transforms, Lightning train/validation, and checkpoint round trip passed')


if __name__=='__main__':
    main()
