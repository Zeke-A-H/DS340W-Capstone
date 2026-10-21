# Local structure check

Checked against README.md on October 20, 2026.

All documented source files and directories exist. GMFlow is initialized at
817fa8decbd5a6de72cd48b72349c0c1097f68cc.

Created the missing dataset and generated-output directories shown in the README.
Restored table and storagefurniture mean-scale statistics to data/ from their
existing TrAssembler checkpoint copies. All three statistics files parse as JSON.

## Dataset still incomplete

The following directories exist but still need their actual downloaded contents:

- data/splits/ (train/test ID files for all three categories)
- data/sym_labels/
- data/llamaft_gt_labels/ (all three categories)
- data/shapenet_partseg/04379243/
- data/shapenet_partseg/storagefurniture/

data/shapenet_partseg/synsetoffset2category.txt is also missing.
data/trassembler_data/*_pkl/ is empty; download processed data or generate it with
`python TrAssembler/dataset.py` after the required input data is available.
Empty output directories are expected before inference.

Existing raw CAD counts: chair 1,029; table 3,324; storagefurniture 312.
Each object directory contains cad.h5. There are 9,328 rendering PNGs (including
masks) and 641 entries in the chair segmentation directory. These counts verify
local presence, not completeness or file integrity against the remote dataset.

To resume the README's dataset download, run from the Img2CAD directory:

```powershell
hf download qq456cvb/img2cad-dataset --repo-type dataset --local-dir data
```

No downloads were started during this structure check.

## Checkpoint paths

Both data/ckpts/hf/ and direct data/ckpts/{llamaft,trassembler,partseg}/
layouts are present. The README downloads into the former, while its usage
examples reference the latter. Existing checkpoints were preserved in place.
Model loading, dependency installation, and training were not tested.
