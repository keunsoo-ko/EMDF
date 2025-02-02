import numpy as np
import torch
import torch.nn.functional as F

def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def load_checkpoint(path, model, optimizer=None, reset_optimizer=True, is_dis=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    if is_dis:
        s = checkpoint["disc"]
    else:
        s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer, checkpoint['global_step'], checkpoint['global_epoch']