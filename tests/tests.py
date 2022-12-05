import sys
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT, 'src'))

from sentiment import SentimentAnalysis
from ner import NerTraining
from bleu import Bleu_Score


print(SentimentAnalysis.test())


print(NerTraining.test())


print(Bleu_Score.test())