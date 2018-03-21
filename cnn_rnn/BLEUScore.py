import numpy as np 
from tqdm import *
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
import h5py
import sys, os
import nltk

class BLEUScore:
    
    def score(self, reference, hypothesis):
        BLEUscore = 0
        for i in range(len(reference)):
            BLEUscore += nltk.translate.bleu_score.sentence_bleu([reference[i]], hypothesis[i])
        
        return BLEUscore / len(hypothesis)