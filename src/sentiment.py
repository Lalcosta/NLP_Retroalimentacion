#Library to get model
from transformers import pipeline
from datasets import load_dataset

#Class to make sentiment analysis with an out of the box model
class SentimentAnalysis:
    
    
    def __init__(self,test):
        self.test=1
        pass
    #Test to verify correc class creation
    def test(self):
        if self.test != 0:
            return "Warm Up class is running"
        else: return "Warm up class is not running"
    
    #Function for sentiment analysis    
    def sentiment_analysis(dset):
        #Model selected beacuse of its capability of analizyng long strings
        sentiment = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
        #Open the dataset
        with open(dset) as data:
            #Go over evey line
            for i in data:
                #Print the result of evaluating the line with the model
                print(sentiment(i)[0]["label"])

    

