import torch


def calc_recall(pred, gt, k):
    s = 0
    for t, y in zip(gt, pred):
        if t in torch.Tensor(y).float()[:k]:
            s += 1
    return s / (1. * len(pred))

