import torch

try:
    # Try with forward slashes
    model = torch.jit.load(r"C:/Other files/GitHub/sapiens/sapiens_lite_host/torchscript/pretrain/checkpoints/sapiens_0.3b_epoch_1600_torchscript.pt2")
    print("Loaded successfully with forward slashes")
except Exception as e:
    print("Forward slash loading failed:", e)