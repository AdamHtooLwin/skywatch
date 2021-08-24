from torch import nn

from .decoder import Decoder
from .resnet import ResNet18Classifier, ResNet18Encoder


class SCAN(nn.Module):
    def __init__(self, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNet18Encoder(pretrained=pretrained)
        self.decoder = Decoder()
        self.clf = ResNet18Classifier(dropout=dropout)

    def forward(self, x):
        outs = self.backbone(x)
        outs = self.decoder(outs)

        s = x + outs[-1]
        clf_out = self.clf(s)

        return outs, clf_out

    def infer(self, x):
        outs, _ = self(x)
        return outs[-1]
