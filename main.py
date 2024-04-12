from transformers import pipeline
import torch
import torch.nn.functional as F

classifier = pipeline("sentiment-analysis")
