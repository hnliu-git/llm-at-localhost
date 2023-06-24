from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from transformers import BertForSequenceClassification

import torch

from torch import autocast

# device = torch.device('mps')

model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased')

print(model)

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print(quantized_model)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

import time

s = time.time()
# with autocast(device_type="cpu", dtype=torch.bfloat16):
print(classifier('hello'))
print(time.time() - s)