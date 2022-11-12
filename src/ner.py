from datasets import load_dataset
from transformers import AutoTokenizer

data=load_dataset("conll2003")

ner_labels = data["train"].features["ner_tags"]
print(ner_labels)