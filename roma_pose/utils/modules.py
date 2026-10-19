from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)



def grad_norm(module: nn.Module) -> float:
    vals = [p.grad.detach().float().norm().square() for p in module.parameters() if p.grad is not None]
    if not vals:
        return 0.0
    return float(torch.sqrt(torch.stack(vals).sum()).detach().cpu())



def set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value



def reset_module_parameters(module: nn.Module) -> None:
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()
