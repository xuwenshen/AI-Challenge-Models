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
from torch.utils import data

class Folder(data.Dataset):
    
    def __init__(self, nb_samples, filepath, is_test):
        
        self.nb_samples = nb_samples
        self.file = h5py.File(filepath)
        self.is_test = is_test
        
    def __getitem__(self, index):
        
        entext = self.file['en'][index]
        enlen = self.file['en_len'][index]
        en_str = self.file['en_str'][index].decode('utf8')
        
        if self.is_test == True:
            return {'entext': entext, 'enlen': enlen, 'enstr':en_str}
        
        
        zhlabel = self.file['label'][index]
        zhlen = self.file['zh_len'][index]
        zhgtruth = self.file['ground_truth'][index]
        zh_str = self.file['zh_str'][index].decode('gb2312')
        
        return {'entext': entext, 'enlen': enlen, 'zhlabel':zhlabel, 'zhlen':zhlen, 'zhgtruth':zhgtruth, 'zhstr':zh_str, 'enstr':en_str}      
        
        
    def __len__(self):
        return self.nb_samples
        
        
