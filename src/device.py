import torch

def available_devices():
    d=["cpu"]
    if torch.cuda.is_available(): d.append("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): d.append("mps")
    return d
