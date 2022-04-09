from copy import deepcopy
from os import path as osp
from typing import Any, Dict, List, Type

import torch
import torch_optimizer as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from torch.nn import Dropout, Linear, Module, ReLU, Sequential
from torch.nn.functional import nll_loss
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import Adam

# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional.classification.f_beta import f1_score
from transformers import AutoModel, RobertaConfig, RobertaForSequenceClassification


class Base(LightningModule):
    def __init__(self, model_name: str = None, local_model: bool = False) -> None:
        super().__init__()
        self.bert_output_size = 768
        print(model_name)
        self.lr = 0.003
        if local_model:
            self.model = RobertaForSequenceClassification(local_model)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        self.classifier = Linear(
            in_features=self.model.config.hidden_size, out_features=1, bias=True
        )
        self.criterion = BCEWithLogitsLoss()
        self.freeze_model(self.model)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, ids, mask):
        out = self.model(ids, attention_mask=mask)
        out = out[0][:, 0, :]
        out = self.classifier(out)
        out = torch.squeeze(out, dim=1)
        return out

    def configure_optimizers(self):
        parameters = list(self.model.parameters()) + list(self.classifier.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = Adam(trainable_parameters, lr=self.lr)
        return optimizer
        # scheduler = ReduceLROnPlateau(optimizer, threshold=5, factor=0.5)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val/val_loss",
        #         "interval": "epoch",
        #     },
        # }

    def training_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        loss = self.criterion(output, labels)
        return loss

    # def training_epoch_end(self, out):
    #     output = torch.stack([x["outputs"] for x in out]).mean()
    #     labels = torch.stack([x["labels"] for x in out]).mean()
    #     f1 = f1_score(preds=output, target=labels)
    #     self.log("train/f1", f1, prog_bar=True)

    def validation_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        loss = self.criterion(output, labels)
        return {"loss": loss, "outputs": torch.sigmoid(output), "labels": labels}

    def validation_epoch_end(self, out):
        loss = torch.stack([x["loss"] for x in out]).mean()
        output = torch.cat([x["outputs"] for x in out])
        labels = torch.cat([x["labels"] for x in out])
        f1 = f1_score(preds=output, target=labels.int())
        self.log("val/val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/val_f1", f1, prog_bar=True, on_epoch=True, on_step=False)

    # def test_step(self, batch, _):
    #     ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
    #     output = self(ids, mask)
    #     f1_score = f1(
    #         sigmoid(output), labels.int(), average="none", num_classes=self.n_classes
    #     )
    #     return {"f1": f1_score}


class DistilRoBERTa(Base):
    def __init__(self, model: str = "distilroberta-base") -> None:
        super().__init__(model)
        print(self.model)
        self.freeze_model(self.model)


class DistilBERT(Base):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.freeze_model(self.model)


class BERT(Base):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.freeze_model(self.model)


class AlBERT(Base):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.freeze_model(self.model)


class RoBERTa(LightningModule):
    def __init__(self, model_name: str = None) -> None:
        super().__init__()
        self.config = RobertaConfig(
            vocab_size=30000,
            max_position_embeddings=514,
            num_attention_heads=4,
            num_hidden_layers=3,
            type_vocab_size=1,
        )
        self.bert_output_size = 768
        print(model_name)
        self.lr = 0.003
        self.model = RobertaForSequenceClassification(config=self.config).roberta
        self.classifier = Linear(
            in_features=self.model.config.hidden_size, out_features=1, bias=True
        )
        self.criterion = BCEWithLogitsLoss()

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, ids, mask):
        out = self.model(ids, attention_mask=mask)
        out = out[0][:, 0, :]
        out = self.classifier(out)
        out = torch.squeeze(out, dim=1)
        return out

    def configure_optimizers(self):
        parameters = list(self.model.parameters()) + list(self.classifier.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = Adam(trainable_parameters, lr=self.lr)
        return optimizer
        # scheduler = ReduceLROnPlateau(optimizer, threshold=5, factor=0.5)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val/val_loss",
        #         "interval": "epoch",
        #     },
        # }

    def training_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        loss = self.criterion(output, labels)
        return loss

    # def training_epoch_end(self, out):
    #     output = torch.stack([x["outputs"] for x in out]).mean()
    #     labels = torch.stack([x["labels"] for x in out]).mean()
    #     f1 = f1_score(preds=output, target=labels)
    #     self.log("train/f1", f1, prog_bar=True)

    def validation_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        loss = self.criterion(output, labels)
        return {"loss": loss, "outputs": torch.sigmoid(output), "labels": labels}

    def validation_epoch_end(self, out):
        loss = torch.stack([x["loss"] for x in out]).mean()
        output = torch.cat([x["outputs"] for x in out])
        labels = torch.cat([x["labels"] for x in out])
        f1 = f1_score(preds=output, target=labels.int())
        self.log("val/val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/val_f1", f1, prog_bar=True, on_epoch=True, on_step=False)

    # def test_step(self, batch, _):
    #     ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
    #     output = self(ids, mask)
    #     f1_score = f1(
    #         sigmoid(output), labels.int(), average="none", num_classes=self.n_classes
    #     )
    #     return {"f1": f1_score}
