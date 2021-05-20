import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_class_accuracy(probs, target, device=None):
    probs = probs.cuda(device)
    target = target.cuda(device)
    assert target.ndim == 1 and target.size(0) == probs.size(0)
    preds = probs.argmax(dim=1)
    accuracy = (preds == target).sum().item() / target.size(0)
    return accuracy, preds
