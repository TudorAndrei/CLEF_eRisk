import logging as log
import os

import pretty_errors
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import logging
from whos_there.callback import NotificationCallback
from whos_there.senders.discord import DiscordSender

from data import DataModule
from models import DistilRoBERTa, KFoldLoop

# logging.set_verbosity_warning()
logging.set_verbosity_error()

log.getLogger("pytorch_lightning").setLevel(log.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# seed_everything(42)

BATCH_SIZE = 1
NW = 8
EPOCHS = 20


if __name__ == "__main__":
    model_name = "distilroberta-base"
    # model_name = "squeezebert/squeezebert-mnli"
    # data = DataModule(
    #     batch_size=BATCH_SIZE,
    #     num_workers=NW,
    #     ground_truth="risk_golden_truth.txt",
    #     folder = "processed"
    # )
    data = DataModule(
        batch_size=BATCH_SIZE,
        num_workers=NW,
        ground_truth="risk_golden_truth_split.txt",
        folder = "split"
    )
    model = DistilRoBERTa(model_name)
    logger = TensorBoardLogger("logs", name=f"Model_{model_name}")
    trainer = Trainer(
        # fast_dev_run=True,
        detect_anomaly=True,
        gpus=1,
        logger=logger,
        max_epochs=EPOCHS,
        callbacks=[
            # ModelCheckpoint(
            #     monitor="val/val_loss",
            #     mode="min",
            #     dirpath=f"models/{model_name}_3",
            #     filename="bert-val_loss{val/val_loss:.2f}",
            #     auto_insert_metric_name=False,
            # ),
            # NotificationCallback(
            #     senders=[
            #         DiscordSender(
            #             webhook_url=web_hook,
            #         )
            #     ]
            # ),
            # LearningRateMonitor(logging_interval="step"),
            # EarlyStopping(monitor="val/val_loss", patience=5),
        ],
    )
    trainer.fit(model, datamodule=data)
