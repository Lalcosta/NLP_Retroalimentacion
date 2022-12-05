import sys
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT, 'src'))

from src.sentiment import SentimentAnalysis
from src.ner import NerTraining
from src.bleu import Bleu_Score

if __name__ == '__main__':
    
    SentimentAnalysis.sentiment_analysis("tiny_movie_reviews_dataset.txt")
    
    bleu=Bleu_Score("english.txt","spanish.txt")
    bleu.limit_hundred_data()
    bleu.bleu()
    
    retraining=NerTraining()
    retraining.retrain()


    
    
    