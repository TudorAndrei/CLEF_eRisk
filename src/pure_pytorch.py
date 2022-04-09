import os
import timeit

import pretty_errors
import torch
from torch.nn import Linear
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

device = torch.device("cuda")

BATCH_SIZE = 8
NW = 8
EPOCHS = 1


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


class Model(torch.nn.Module):
    def __init__(self, model_name=None) -> None:
        super().__init__()
        print(model_name)
        self.lr = 0.003
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


def train(model, optimizer, loss_func, train_dl, epochs):
    for _ in tqdm(range(epochs)):
        start_epoch = timeit.timeit()
        for batch in tqdm(train_dl):
            ids, mask, labels = batch["ids"], batch["mask"], batch["labels"]
            optimizer.zero_grad()
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            output = model(ids, mask)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
        end_epoch = timeit.timeit()
        print("epoch")
        print(end_epoch - start_epoch)
        break


if __name__ == "__main__":
    model_name = "distilroberta-base"
    data = BaseDataset(
        model_name=model_name,
        ground_truth="risk_golden_truth_chunks.txt",
        processed_folder="chunked",
    )
    dl = DataLoader(data, batch_size=BATCH_SIZE, num_workers=NW, pin_memory=True)

    loss_func = BCEWithLogitsLoss()
    model = Model(model_name)
    model.to(device)
    optimizer = Adam(
        model.classifier.parameters(),
        # model.classifier.parameters()
        lr=0.003,
    )
    train(model, optimizer, loss_func, dl, EPOCHS)
