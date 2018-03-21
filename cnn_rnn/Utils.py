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

class Utils:
    
    def __init__(self, batch_size, nb_samples, en_path, is_test, zh_path = None):
        
        self.current_batch = 0
        self.batch_size = batch_size
        self.nb_samples = nb_samples
        self.en_file = h5py.File(en_path)
        self.zh_file = h5py.File(zh_path)
        self.is_test = is_test
        self.shuffled_id = np.arange(nb_samples)
        random.shuffle(self.shuffled_id)
        
        
    def next_batch(self):
        
        start = self.current_batch * self.batch_size
        end = min(start+self.batch_size, self.nb_samples)
        self.current_batch += 1 
        
        ids = self.shuffled_id[start:end]
        ids = sorted(ids)
        
  
        is_again = False
        if end == self.nb_samples:
            is_again = True
            random.shuffle(self.shuffled_id)
            self.current_batch = 0
        
        entext = self.en_file['en'][ids]
        enlen = self.en_file['length'][ids]
        
        tenlen = enlen.reshape((len(enlen), 1))
        entext = np.hstack((entext, tenlen))
        entext = sorted(entext, reverse=True, key=lambda entext:entext[-1])
        entext = np.array(entext)[:, :-1]
        
        if self.is_test == True:
            return {'entext': entext, 'enlen': enlen, 'flag':is_again}
        
        
        zhlabel = self.zh_file['label'][ids]
        zhlen = self.zh_file['length'][ids]
        zhgtruth = self.zh_file['ground_truth'][ids]
        
        return {'entext': entext, 'enlen': enlen, 'zhlabel':zhlabel, 'zhlen':zhlen, 'zhgtruth':zhgtruth, 'flag':is_again}
        
        
