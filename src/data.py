import os
import os.path as osp
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class TXTDataset(Dataset):
    def __init__(self, path_to_data="data", model_name="distilroberta-base"):
        self.labels = []
        self.path_to_data = path_to_data
        self.ext = ".txt"
        self.path_to_processed = os.path.join(self.path_to_data, "processed")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        path_labels = os.path.join(path_to_data, "risk_golden_truth.txt")
        with open(path_labels, "r") as f:
            for line in f:
                subject, label = line.split()
                self.labels.append((subject, label))

    def __getitem__(self, idx):
        subject, label = self.labels[idx]
        subject_path = os.path.join(self.path_to_processed, subject) + self.ext
        with open(subject_path, "r") as f:
            text = " ".join(f.readlines())
        text = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        return {
            "ids": torch.tensor(text["input_ids"], dtype=torch.long),
            "mask": torch.tensor(text["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(int(label), dtype=torch.float),
        }

    def __len__(self):
        return len(self.labels)


class DataModule(LightningDataModule):
    def __init__(
        self,
        model: str = None,
        num_workers: int = 8,
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.model = model

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = TXTDataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.dpm_val,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )
