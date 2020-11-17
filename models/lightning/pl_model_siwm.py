from torch import nn
from catalyst.contrib.nn.criterion.focal import FocalLossMultiClass
import pytorch_lightning as pl

from models.scan import SCAN


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = SCAN()
        self.log_cues = not self.hparams.cue_log_every == 0
        if self.hparams.use_focal_loss:
            self.clf_criterion = FocalLossMultiClass()
        else:
            self.clf_criterion = nn.CrossEntropyLoss()

    def get_progress_bar_dicts(self):
        items = super().get_progress_bar_dicts()
        return items

    def forward(self, x):
        return self.model(x)

    def infer(self, x):
        outs, _ = self.model(x)
        return outs[-1]

    def classify(self, x):
        outs, clf_out = self.model(x)
        return outs[-1], clf_out
