import torch
from pytorch_lightning import LightningModule
from torch.nn import Dropout, Linear, Module, ReLU, Sequential, Sigmoid
from torch.nn.functional import sigmoid
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.classification.f_beta import f1_score
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification


class Base(LightningModule):
    def __init__(self, model_name=None, n_classes: int = 7, hidden_size=1024) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.bert_output_size = 768
        self.hidden_size = hidden_size
        print(model_name)
        print(self.hidden_size)

        self.lr = 0.003
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.hidden = Sequential(
            Linear(self.bert_output_size, self.hidden_size), ReLU(), Dropout(0.1)
        )
        self.classifier = Module()
        self.criterion = BCEWithLogitsLoss()

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask)
        out = out[0]
        out = out[:, 0]
        out = self.hidden(out)
        out = self.classifier(out)
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.classifier.parameters(), lr=self.lr)
        # scheduler = ReduceLROnPlateau(optimizer, threshold=5, factor=0.5)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val/val_loss",
            #     "interval": "epoch",
            # },
        }

    def training_step(self, batch, _):
        ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
        output = self(ids, mask)
        output = torch.squeeze(output)
        loss = self.criterion(output, labels)
        f1 = f1_score(output, labels.int())
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return {"loss": loss, "f1": f1}

    def training_epoch_end(self, out):
        f1_score = torch.stack([x["f1"] for x in out]).mean()
        self.log("train/f1", f1_score, prog_bar=True, on_epoch=True, on_step=False)

    # def validation_step(self, batch, _):
    #     ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
    #     output = self(ids, mask)
    #     loss = self.criterion(output, labels)
    #     f1_score = f1(
    #         sigmoid(output), labels.int(), average="macro", num_classes=self.n_classes
    #     )
    #     return {"loss": loss, "f1": f1_score}

    # def validation_epoch_end(self, out):
    #     loss = torch.stack([x["loss"] for x in out]).mean()
    #     f1_score = torch.stack([x["f1"] for x in out]).mean()
    #     self.log("val/val_loss", loss, on_epoch=True, on_step=False)
    #     self.log("val/val_f1", f1_score, on_epoch=True, on_step=False)

    # def test_step(self, batch, _):
    #     ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
    #     output = self(ids, mask)
    #     f1_score = f1(
    #         sigmoid(output), labels.int(), average="none", num_classes=self.n_classes
    #     )
    #     return {"f1": f1_score}


class DistilRoBERTa(Base):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = self.model.roberta
        self.freeze_model(self.model)
        self.classifier = Linear(
            in_features=self.hidden_size, out_features=1, bias=True
        )
        print(self.classifier)
