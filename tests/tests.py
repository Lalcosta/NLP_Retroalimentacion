import sys
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT, 'src'))

from Warm_up_SA import WarmUp
from nerclass import NerTraining
from bleu import Bleu_Score


print(WarmUp.test())


print(NerTraining.test())


print(Bleu_Score.test())