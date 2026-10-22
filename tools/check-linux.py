"""Exercise the GPU, CUDA rasterizer, CAD kernel, and inference imports."""
import json
from pathlib import Path
import runpy
import sys

import torch
import nvdiffrast.torch as dr
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from huggingface_hub import get_token, hf_hub_download

assert torch.cuda.is_available(), "No CUDA GPU in this session"
x = torch.ones(32, 32, device="cuda")
assert (x @ x).mean().item() == 32
ctx = dr.RasterizeCudaContext()
vertices = torch.tensor([[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]]], device="cuda", dtype=torch.float32)
triangles = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
raster, _ = dr.rasterize(ctx, vertices, triangles, resolution=[32, 32])
assert raster[..., 3].sum().item() > 0
assert not BRepPrimAPI_MakeBox(1, 1, 1).Shape().IsNull()
for script in ["TrAssembler/eval.py", "LlamaFT/infer.py"]:
    sys.path.insert(0, str(Path(script).parent.resolve()))
    runpy.run_path(script, run_name="dependency_check")
report = {
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "cuda_rasterizer": "passed",
    "cad_kernel": "passed",
    "inference_imports": "passed",
    "huggingface_token_available": bool(get_token()),
    "stage1_h5_count": len(list(Path("data/output/llamaft_h5/chair").glob("*.h5"))),
}
try:
    hf_hub_download("meta-llama/Llama-3.2-11B-Vision-Instruct", "config.json")
    report["llama_access"] = "passed"
except Exception as exc:
    report["llama_access"] = type(exc).__name__
Path("logs/linux-readiness.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
