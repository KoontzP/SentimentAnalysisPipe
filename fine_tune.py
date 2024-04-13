from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load pre-trained model
model_name = "distilbert-base-uncased"


# Create text and labels from data
def read_data_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels


train_texts, train_labels = read_data_split('train_data_path')
test_texts, test_labels = read_data_split('test_data_path')


train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, test_labels, test_size=.2)


class PytorchDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# ensure that all of out sequences are padded to the same length and are truncated to be no longer than model's
# maximum input length. This will allow us to feed batches of sequences into the model at the same time.
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = PytorchDataset(train_encodings, train_labels)
val_dataset = PytorchDataset(val_encodings, val_labels)
test_dataset = PytorchDataset(test_encodings, test_labels)

# arguments/hyperparameters for training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,                   # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained(model_name)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


## native PyTorch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

num_train_epochs = 2
for epoch in range(num_train_epochs):
    for batch in train_loader:
        optim.zero_grad()

        # push to device for GPU cuda
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['labels'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
