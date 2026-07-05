"""
model.py — transfer-learning regressor.

A pretrained ResNet18 backbone (ImageNet) with the classifier replaced by
a single linear output = predicted age. Regression keeps the threshold a
free knob, so the challenge-age buffer is applied at eval time, not baked
into the model. Swap to mobilenet_v3_small for a lighter/faster model.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(backbone="resnet18", pretrained=True):
    if backbone == "resnet18":
        net = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, 1)
    elif backbone == "mobilenet":
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, 1)
    else:
        raise ValueError(backbone)
    return net
