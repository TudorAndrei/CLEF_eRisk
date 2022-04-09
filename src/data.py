import os
from dataclasses import dataclass
from typing import Optional, Type

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset
from transformers import RobertaTokenizerFast
from transformers.models.auto.tokenization_auto import AutoTokenizer


class BaseDataset(Dataset):
    def __init__(
        self,
        path_to_data: str = "data",
        model_name: str = "distilroberta-base",
        ground_truth: str = "risk_golden_truth.txt",
        processed_folder: str = "processed",
        roberta_pretrained_path: str = None,
    ):
        self.labels = []
        self.ext = ".txt"

        self.path_to_data = path_to_data
        self.path_to_processed = os.path.join(self.path_to_data, processed_folder)
        self.path_labels = os.path.join(self.path_to_data, ground_truth)

        if roberta_pretrained_path:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                roberta_pretrained_path
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        with open(self.path_labels, "r") as f:
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
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            # return_tensors="pt",
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
        num_workers: int = 8,
        batch_size: int = 2,
        model_name: str = "distilroberta-base",
        ground_truth: str = "risk_golden_truth_split.txt",
        folder: str = "split",
        roberta_pretrained_path: str = None,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.ground_truth = ground_truth
        self.folder = folder
        self.roberta_pretrained_path = roberta_pretrained_path

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = BaseDataset(
            model_name=self.model_name,
            ground_truth=self.ground_truth,
            processed_folder=self.folder,
            roberta_pretrained_path=self.roberta_pretrained_path,
        )
        size = len(dataset)
        train_size = int(size * 0.8)
        self.train_dataset, self.test_dataset = random_split(
            dataset, [train_size, size - train_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
