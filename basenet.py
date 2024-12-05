import torch.nn as nn
import torch

class ResClassifier(nn.Module):
    def __init__(self, num_classes=7, num_unit=128, middle=64):
        super(ResClassifier, self).__init__()
        layers = []

        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x)
        return x
