"""Small preflight check for the Kaggle GPU environment."""
import importlib
import sys

import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is unavailable. Enable GPU in Kaggle Notebook Settings > Accelerator.")

for name in ["torchvision", "sentence_transformers", "transformers", "faiss", "qwen_vl_utils"]:
    mod = importlib.import_module(name)
    print(f"{name}: OK", getattr(mod, "__version__", ""))
