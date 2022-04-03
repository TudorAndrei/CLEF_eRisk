import os
from dataclasses import dataclass
from typing import Optional, Type

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class BaseDataset(Dataset):
    def __init__(
        self,
        path_to_data="data",
        model_name="distilroberta-base",
        ground_truth="risk_golden_truth.txt",
        processed_folder="processed",
    ):
        self.labels = []
        self.ext = ".txt"

        self.path_to_data = path_to_data
        self.path_to_processed = os.path.join(self.path_to_data, processed_folder)
        self.path_labels = os.path.join(self.path_to_data, ground_truth)

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
            return_token_type_ids=True,
            # return_tensors='pt',
            # return_attention_mask=True,
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
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.ground_truth = ground_truth
        self.folder = folder

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = BaseDataset(
            model_name=self.model_name,
            ground_truth=self.ground_truth,
            processed_folder=self.folder,
        )

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


@dataclass
class MNISTKFoldDataModule(LightningDataModule):
    train_dataset: Optional[Type[Dataset]] = None
    test_dataset: Optional[Type[Dataset]] = None
    train_fold: Optional[Type[Dataset]] = None
    val_fold: Optional[Type[Dataset]] = None
    num_workers: int = 8
    batch_size: int = 32

    def prepare_data(self) -> None:
        dataset = TXTDataset()
        size = len(dataset)
        train_size = int(size * 0.8)
        self.train_dataset, self.test_dataset = random_split(
            dataset, [train_size, size - train_size]
        )

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in KFold(num_folds).split(range(len(self.train_dataset)))
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_fold, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_fold, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)

    def __post_init__(cls):
        super().__init__()
