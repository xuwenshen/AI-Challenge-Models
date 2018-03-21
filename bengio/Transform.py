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
from collections import defaultdict
import nltk
import string


def revers_dic(dic):
    
    rdic = defaultdict()
    for (k,v) in dic.items():
        rdic[v] = k
    rdic = dict(rdic)
    return rdic
        
    
class Transform:
    
    def __init__(self, zh_voc_path, en_voc_path):
        
        self.zh_voc = json.load(open(zh_voc_path))
        self.en_voc = json.load(open(en_voc_path))
        self.zh_rvoc = revers_dic(self.zh_voc)
        self.en_rvoc = revers_dic(self.en_voc)
        
        self.zh_go_id = self.zh_voc['go#']
        self.zh_eos_id = self.zh_voc['eos#']
        self.zh_pad_id = self.zh_voc['pad#']
        self.zh_nuk_id = self.zh_voc['nuk#']
        
        
        self.en_go_id = self.en_voc['go#']
        self.en_eos_id = self.en_voc['eos#']
        self.en_pad_id = self.en_voc['pad#']
        self.en_nuk_id = self.en_voc['nuk#']
        
        
        self.go = 'go#'
        self.eos = 'eos#'
        self.pad = 'pad#'
        self.nuk = 'nuk#'
        

    def clip(self, lst, language):
        
        assert language == 'en' or language == 'zh'
        ids = []
        
        if language == 'en':
            end = 0
            for i in range(len(lst)):
                if lst[i] == self.en_pad_id:
                    end = i
                    break
            ids = lst[:end]
        
        else:
            
            end = 0
            for i in range(len(lst)):
                if lst[i] == self.zh_eos_id:
                    end = i
                    break
            ids = lst[:end]
            
        
        return ids
            
        
        
    def i2t(self, lst, language):
        
        assert language == 'en' or language == 'zh'
 
        text = ''
        
        if language == 'en':
   
            for i in range(len(lst)):
                
                token = self.en_rvoc[lst[i]]
                if token in string.punctuation:
                    text += self.en_rvoc[lst[i]]
                else:
                    text += ' ' + self.en_rvoc[lst[i]]
                
            text = text.lstrip(' ')
        
        else:

            for i in range(len(lst)):
                text += self.zh_rvoc[lst[i]]
            
        
        return text