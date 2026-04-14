from typing import Literal

import torch.nn as nn
from torchvision import models


def build_resnet50(
    num_classes: int,
    unfreeze_blocks: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if unfreeze_blocks > 0:
        blocks = [model.layer4, model.layer3, model.layer2, model.layer1]
        for block in blocks[: min(unfreeze_blocks, len(blocks))]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def resolve_device(device: Literal["auto", "cpu", "cuda"] = "auto") -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
