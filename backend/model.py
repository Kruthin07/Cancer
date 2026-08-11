import torch.nn as nn
from torchvision import models

class CancerCNN(nn.Module):
    def __init__(self, trainable_layers: int = 1, pretrained: bool = True):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        for param in self.model.parameters():
            param.requires_grad = False

        # Fine-tune last N ResNet stages (1 => layer4 only, 2 => layer3+layer4, ...)
        trainable_layers = max(0, min(int(trainable_layers), 4))
        if trainable_layers >= 1:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        if trainable_layers >= 2:
            for param in self.model.layer3.parameters():
                param.requires_grad = True
        if trainable_layers >= 3:
            for param in self.model.layer2.parameters():
                param.requires_grad = True
        if trainable_layers >= 4:
            for param in self.model.layer1.parameters():
                param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)  # LOGITS
        )

    def forward(self, x):
        return self.model(x)