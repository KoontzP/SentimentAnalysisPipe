from transformers import pipeline
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

x_train = ["Hello", "Great news, you are cancer free"]

batch = tokenizer(x_train, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    outputs = model(**batch)
    label_ids = torch.argmax(outputs.logits, dim=1)
    print(label_ids)
    labels = [model.config.id2label[label_ids] for label_ids in label_ids.tolist()]
    print(labels)

