import torch.nn as nn
from models.sync_batchnorm import SynchronizedBatchNorm2d


def BatchNorm(planes, sync_bn=False):
    if not sync_bn:
        return SynchronizedBatchNorm2d(planes)
    return nn.BatchNorm2d(planes)


def initial_weight(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
