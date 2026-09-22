"""Two-branch CNN + azimuth metadata fusion model."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class LunarFusionModel(nn.Module):
    def __init__(self, pretrained: bool = True, metadata_dropout: float = 0.1,
                 classifier_dropout: float = 0.3, channels: int = 1, use_metadata: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        old_conv = backbone.conv1
        if channels not in (1, 3):
            raise ValueError('channels must be 1 or 3')
        if channels == 1:
            backbone.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                       stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.use_metadata = use_metadata
        image_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_branch = backbone

        self.metadata_branch = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(inplace=True),
            nn.Dropout(metadata_dropout), nn.Linear(32, 16), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_features + 16, 128), nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout), nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, azimuth: torch.Tensor) -> torch.Tensor:
        image_features = self.image_branch(image)
        metadata_features = self.metadata_branch(azimuth) if self.use_metadata else image_features.new_zeros((len(image), 16))
        return self.classifier(torch.cat([image_features, metadata_features], dim=1)).squeeze(1)
