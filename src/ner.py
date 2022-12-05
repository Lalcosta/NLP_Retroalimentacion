from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

#Class for Re-training a NER model
class NerTraining:
    def __init__(self):
        #Dataset
        self.data=load_dataset("conll2003") 
        #Tokenizer
        self.tokenizer= BertTokenizerFast.from_pretrained("bert-base-uncased") 
        #Model for re-train
        self.model= AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)
        #If you have a gpu you can use this for training
        self.torch_device='cuda' if torch.cuda.is_available() else 'cpu'
        #Send model to gpu
        self.model = self.model.to(self.torch_device)
        #Training arguments
        self.args = TrainingArguments( 
                                    "test-ner",
                                    evaluation_strategy = "epoch", 
                                    learning_rate=2e-5, 
                                    per_device_train_batch_size=8, 
                                    per_device_eval_batch_size=8, 
                                    num_train_epochs=10, 
                                    weight_decay=0.01, 
                                    ) 
    #Test to verify data is stored
    def test(self):
        if len(self.data)!=0:
            return "NER Trainer data is loaded"
        else: return "Data not loaded"
        
    #Function that tokenize and align labels of the conll2003 dataset, in order to train the bert-base-uncased model
    def conll2003_tokenizer(self, data, labeled=True): 
        self.tokens = self.tokenizer(data["tokens"], truncation=True, is_split_into_words=True) 
        labels = [] 
        for i, label in enumerate(data["ner_tags"]): 
            word_ids = self.tokens.word_ids(batch_index=i) 
            # word_ids() => Return a list mapping the tokens
            # to their actual word in the initial sentence.
            # It Returns a list indicating the word corresponding to each token. 
            prev_id = None 
            label_ids = []
            # Special tokens like `` and `<\s>` are originally mapped to None 
            # We need to set the label to -100 so they are automatically ignored in the loss function.
            for k in word_ids: 
                if k is None: 
                    # set –100 as the label for these special tokens
                    label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the labeled flag.
                elif k != prev_id:
                    # if current k is != prev then its the most regular case
                    # and add the corresponding token                 
                    label_ids.append(label[k]) 
                else: 
                    # to take care of sub-words which have the same k
                    # set -100 as well for them, but only if labeled == False
                    label_ids.append(label[k] if labeled else -100) 
                    # mask the subword representations after the first subword
                    
                prev_id = k 
            labels.append(label_ids) 
        self.tokens["labels"] = labels 
        return self.tokens 

    
    #Metric to compute during training
    def compute_metrics(self, eval_preds): 
        self.metric = load_metric("seqeval") 
        self.label_list = self.data["train"].features["ner_tags"].feature.names
        pred_logits, labels = eval_preds 
        
        pred_logits = np.argmax(pred_logits, axis=2) 
        # the logits and the probabilities are in the same order,
        # so we don’t need to apply the softmax
        
        # We remove all the values where the label is -100
        predictions = [ 
            [self.label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(pred_logits, labels) 
        ] 
        #For testing during training
        true_labels = [ 
        [self.label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
        ] 
        results = self.metric.compute(predictions=predictions, references=true_labels) 
        return { 
                "precision": results["overall_precision"], 
                "recall": results["overall_recall"], 
                "f1": results["overall_f1"], 
                "accuracy": results["overall_accuracy"], 
                } 

    #Function to re-train model
    def retrain(self):
        #Use tokenized datasets
        self.tokenized_datasets = self.data.map(self.conll2003_tokenizer, batched=True)
        data_collator = DataCollatorForTokenClassification(self.tokenizer) 
        
        #Creating the trainer with model, training arguments, tokenized train dataset, 
        #tokenized test dataset, tokenizer and metrics to compute
        trainer = Trainer( 
            self.model, 
            self.args, 
        train_dataset=self.tokenized_datasets["train"], 
        eval_dataset=self.tokenized_datasets["validation"], 
        data_collator=data_collator, 
        tokenizer=self.tokenizer, 
        compute_metrics=self.compute_metrics 
        ) 
        #Re-train model
        trainer.train()

        #After training computed metrics were printed so I created a csv file with those metrics
        #Acces csv file
        loss=pd.read_csv("errors.csv")
        #Plot train loss
        plt.plot(loss["Epoch"],loss["Train_loss"], label="Train loss")
        #Plot test loss
        plt.plot(loss["Epoch"],loss["Ev_loss"], label="Train errors")
        plt.title("Test error vs train error")
        plt.show()
        
