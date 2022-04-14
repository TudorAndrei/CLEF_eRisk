# %%
from torch.utils.data import DataLoader, Dataset
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig, RobertaTokenizerFast
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda")


# %%
path_to_data="data"
model_name="distilroberta-base"
ground_truth="risk_golden_truth.txt"
processed_folder="processed"

labels = []
ext = ".txt"
path_to_processed = os.path.join(path_to_data, processed_folder)
path_labels = os.path.join(path_to_data, ground_truth)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
with open(path_labels, "r") as f:
    for line in f:
        subject, label = line.split()
        labels.append((subject, label))
    # print(labels)

# %%
subject, label = labels[0]
subject_path = os.path.join(path_to_processed, subject) + ext
print(subject_path)

# %%
# with open(subject_path, "r") as f:
#     text = " ".join(f.readlines())
# text = tokenizer.encode(
#     text,
#     add_special_tokens=True,
#     # max_length=512,
#     padding=True,
#     return_tensors='pt',
#     truncation=True,
# )
# text


# %%
class BaseDataset(Dataset):
    def __init__(
        self,
        path_to_data="data",
        model_name="distilroberta-base",
        ground_truth="risk_golden_truth_chunks.txt",
        processed_folder="chunked",
    ):
        self.labels = []
        self.ext = ".txt"

        self.path_to_data = path_to_data
        self.path_to_processed = os.path.join(self.path_to_data, processed_folder)
        self.path_labels = os.path.join(self.path_to_data, ground_truth)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
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
            "ids": torch.tensor(text['input_ids'], dtype=torch.float),
            "mask": torch.tensor(text["attention_mask"], dtype=torch.float),
            "labels": torch.tensor(int(label), dtype=torch.float),
        }

    def __len__(self):
        return len(self.labels)


dl = DataLoader(BaseDataset(), batch_size=4)

# %%
# config = AutoConfig.from_pretrained('distilroberta-base', num_labels=1)
model = AutoModel.from_pretrained('distilroberta-base').embeddings

# %%
print(AutoModel.from_pretrained('distilroberta-base'))

# %%
print(model)

# %%
for x in dl:
    ids,mask, labels = x['ids'],x['mask'], x['labels']
    out = model(ids)
    print(out)
    break

# %%
dl


