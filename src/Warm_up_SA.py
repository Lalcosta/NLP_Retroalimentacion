#Library to get model
from transformers import pipeline
from datasets import load_dataset
class WarmUp:
    
    def __init__(self):
        pass

    def sentiment_analysis(dset):
        #Model selected beacuse of its capability of analizyng long strings
        sentiment = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
        #Open the dataset
        with open(dset) as data:
            #Go over evey line
            for i in data:
                #Print the result of evaluating the line with the model
                print(sentiment(i)[0]["label"])

class NerTraining:
    
    def __init__(self):
        pass
    
    
    
WarmUp.sentiment_analysis("tiny_movie_reviews_dataset.txt")
