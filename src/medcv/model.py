"""Modelos pequeños, entrenados desde cero (sin pesos preentrenados).

Entrenar desde cero evita depender de descargas y de pesos con datos de
origen no controlado; es una línea base honesta, no un modelo de estado del arte.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_channels, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(arch: str, in_channels: int, n_classes: int) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(in_channels, n_classes)
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights=None, num_classes=n_classes)
        # Imágenes pequeñas: convolución inicial 3x3 sin reducción ni maxpool.
        m.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    raise ValueError(f"Arquitectura desconocida: {arch}")


def last_conv_layer(model: nn.Module, arch: str) -> nn.Module:
    """Capa objetivo para Grad-CAM."""
    if arch == "simple_cnn":
        return model.features[-1][0]  # última convolución
    if arch == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"Arquitectura desconocida: {arch}")
