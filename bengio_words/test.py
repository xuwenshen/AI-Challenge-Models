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
from torch.utils.data import DataLoader
import time
import math, re
from torch.nn import utils

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from Utils import Utils
from net import Seq2Seq
from hyperboard import Agent
from BLEUScore import BLEUScore
from Folder import Folder
from Transform import Transform

bleuscore = BLEUScore()
model_dir = '/data/xuwenshen/ai_challenge/code/bengio_words/models/'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


def test(net, test_loader, transform, batch_size):

    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    all_pre = []
    all_en = []
    all_en = []
    for (_, data) in enumerate(test_loader, 0):


        entext = data['entext']
        enlen = data['enlen']
        enstr = data['enstr']
            
        zhgtruth = torch.Tensor([[transform.zh_go_id for j in range(51)] for i in range(entext.size(0))])

        logits, predic = net(entext, zhgtruth, enlen, 0, False)

        all_pre.extend(predic)
        all_en.extend(entext.tolist())
        
        del logits, predic

        
    fout = open('/data/xuwenshen/ai_challenge/code/bengio_words/pred.txt', 'w')
    for i in range(len(all_pre)):
        all_pre[i] = transform.clip(all_pre[i], language='zh')
        all_en[i] = transform.clip(all_en[i], language='en')
        
        zh_gen = transform.i2t(all_pre[i], language='zh')
        zh_gen = re.sub(r'nuk#', '', zh_gen)
        fout.write(zh_gen + '\n')
            
    fout.close()
    

    
    

if __name__ == '__main__':
    
    
    batch_size = 64
    nb_samples = 8000
    path = '/data/xuwenshen/ai_challenge/data/valid/valid_words/valid.h5py'
    
    test_folder = Folder(filepath=path,
                         is_test=True,
                         nb_samples=nb_samples)
    test_loader = DataLoader(test_folder,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=False)
    
    
    en_voc_path = '/data/xuwenshen/ai_challenge/data/train/train_words/en_voc.json'
    zh_voc_path = '/data/xuwenshen/ai_challenge/data/train/train_words/zh_voc.json'
    transform = Transform(en_voc_path=en_voc_path,
                         zh_voc_path=zh_voc_path)

    
    en_dims = 256
    en_voc = len(transform.en_voc)
    zh_dims = 256
    zh_voc = len(transform.zh_voc)
    dropout = 0.5
    en_hidden = 256
    zh_hidden = 400
    atten_vec_size = 712
    
    net = Seq2Seq(en_dims=en_dims, 
                  zh_dims=zh_dims,
                  dropout=dropout,
                  en_hidden=en_hidden, 
                  zh_hidden=zh_hidden,
                  atten_vec_size=atten_vec_size,
                  entext_len=60)
    
    pre_trained = torch.load('/data/xuwenshen/ai_challenge/code/bengio_words/models/ssprob-0.817099-loss-6.122144-score-0.522474-steps-400-model.pkl') 
    net.load_state_dict(pre_trained)
    print (net)
   
    
    test(test_loader=test_loader,
          transform=transform,
          net=net,
          batch_size=batch_size)
    
