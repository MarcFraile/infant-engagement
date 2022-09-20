import torch
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    """
    Encoder for Net, provided by `torchvision.models`.

    * Mixed-convolutions model (mc3_18), pretrained.
    """

    def __init__(self):
        super().__init__()
        self.base_model = models.video.mc3_18(pretrained=True)
        self.base_model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class Classifier(nn.Module):
    """
    Classifier head for Net.

    * Simple FC layer (512 -> 2).
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=512, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        return y


class Net(nn.Module):
    """
    Binary video classifier.

    * Torchvision-provided, pre-trained mixed-convolutions model (mc3_18) with head swapped for a 2-class (true/false) classifier.
    * Returns raw scores.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        score = self.classifier(h)
        return score
