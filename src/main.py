import glob
import logging as log
import os

import pretty_errors
from dotenv import dotenv_values
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import logging
from whos_there.callback import NotificationCallback
from whos_there.senders.discord import DiscordSender

from data import DataModule
from models import Transformer

# logging.set_verbosity_warning()
logging.set_verbosity_error()
config = dotenv_values(".env")

# log.getLogger("pytorch_lightning").setLevel(log.WARNING)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# seed_everything(42)

BATCH_SIZE = 2
NW = 8
EPOCHS = 1
web_hook = config["DISCORD_WEBHOOK"]


if __name__ == "__main__":
    model_name = "roberta"

    tokenizer_path = r"models/roberta-tokenizer"
    params = {
        "batch_size": BATCH_SIZE,
        "num_workers": NW,
        "model_name": model_name,
        "roberta_pretrained_path": tokenizer_path,
    }
    data = DataModule(
        ground_truth="risk_golden_truth_chunks.txt", folder="chunked", **params
    )

    model = Transformer()
    trainer = Trainer(
        # fast_dev_run=True,
        detect_anomaly=True,
        # gpus=1,
        logger=TensorBoardLogger("logs", name=f"model_{model_name}"),
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
            # # LearningRateMonitor(logging_interval="step"),
            # EarlyStopping(monitor="val/val_loss", patience=5),
        ],
    )
    trainer.fit(model, data)
