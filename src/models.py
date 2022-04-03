from copy import deepcopy
from os import path as osp
from typing import Any, Dict, List, Type

import torch
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
from transformers import AutoModel


class Base(LightningModule):
    def __init__(self, model_name=None) -> None:
        super().__init__()
        self.bert_output_size = 768
        print(model_name)
        self.lr = 0.003
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = Linear(in_features=768, out_features=1, bias=True)
        self.criterion = BCEWithLogitsLoss()

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, ids, mask):
        out = self.model(ids, attention_mask=mask)
        out = out[0][:, 0, :]
        out = self.classifier(out)
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
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
        output = torch.squeeze(output, dim=1)
        loss = self.criterion(output, labels)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    # def training_epoch_end(self, out):
    #     output = torch.stack([x["outputs"] for x in out]).mean()
    #     labels = torch.stack([x["labels"] for x in out]).mean()
    #     f1 = f1_score(preds=output, target=labels)
    #     self.log("train/f1", f1, prog_bar=True)

    # def validation_step(self, batch, _):
    #     ids, mask, labels = batch[""], batch["labels"]
    #     output = self(ids, mask)
    #     # output = torch.squeeze(output, 1)
    #     loss = self.criterion(output, labels)
    #     f1 = f1_score(output, labels.int())
    #     return {"loss": loss, "f1": f1}

    # def validation_epoch_end(self, out):
    #     loss = torch.stack([x["loss"] for x in out]).mean()
    #     f1 = torch.stack([x["f1"] for x in out]).mean()
    #     self.log("val/val_loss", loss, on_epoch=True, on_step=False)
    #     self.log("val/val_f1", f1, on_epoch=True, on_step=False)

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
        # self.freeze_model(self.model)


class SqueezeBERT(Base):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.freeze_model(self.model)


class EnsembleVotingModel(LightningModule):
    def __init__(
        self, model_cls: Type[LightningModule], checkpoint_paths: List[str]
    ) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )
        self.test_acc = Accuracy()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = nll_loss(logits, batch[1])
        self.test_acc(logits, batch[1])
        self.log("test/test_acc", self.test_acc)
        self.log("test/test_loss", loss)


class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(
            self.trainer.lightning_module.state_dict()
        )

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(
            osp.join(self.export_path, f"model.{self.current_fold}.pt")
        )
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [
            osp.join(self.export_path, f"model.{f_idx + 1}.pt")
            for f_idx in range(self.num_folds)
        ]
        voting_model = EnsembleVotingModel(
            type(self.trainer.lightning_module), checkpoint_paths
        )
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]
