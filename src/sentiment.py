#Library to get model
from transformers import pipeline
from datasets import load_dataset
class SentimentAnalysis:
    
    def __init__(self,test):
        self.test=1
        pass

    def test(self):
        if self.test != 0:
            return "Warm Up class is running"
        else: return "Warm up class is not running"
        
    def sentiment_analysis(dset):
        #Model selected beacuse of its capability of analizyng long strings
        sentiment = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
        #Open the dataset
        with open(dset) as data:
            #Go over evey line
            for i in data:
                #Print the result of evaluating the line with the model
                print(sentiment(i)[0]["label"])

    

