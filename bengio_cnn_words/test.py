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
from torch.utils import data
from torch.nn import utils
from torch.utils.data import DataLoader
import time
import math, re

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from net import Seq2Seq
from Transform import Transform
from BLEUScore import BLEUScore
from hyperboard import Agent
from Folder import Folder

    

def test(net,test_loader,transform, batch_size):


    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    all_pre = []
    all_en = []
    all_en = []
    for (_, data) in enumerate(test_loader, 0):

        entext = data['entext']
        enlen = data['enlen']
        zhgtruth = torch.Tensor([[transform.zh_go_id for j in range(51)] for i in range(entext.size(0))])

        logits, predic = net(entext, zhgtruth, enlen, 0, False)

        all_pre.extend(predic)
        all_en.extend(entext.tolist())
        
        del logits, predic

    fout = open('/data/xuwenshen/ai_challenge/code/bengio_cnn_words/pred.txt', 'w')
    for i in range(len(all_pre)):
        all_pre[i] = transform.clip(all_pre[i], language='zh')
        #all_en[i] = transform.clip(all_en[i], language='en')
        
        #en_text = transform.i2t(all_en[i], language='en')
        zh_gen = transform.i2t(all_pre[i], language='zh')
        zh_gen = re.sub(r'nuk#', '', zh_gen)
        fout.write(zh_gen + '\n')
            
    fout.close()
    

if __name__ == '__main__':
    
    
    batch_size = 32
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
    zh_dims = 256
    dropout = 0.5
    en_hidden = 256
    zh_hidden = 400
    atten_vec_size = 712
    channels = 1024
    kernel_size = 1
    
    net = Seq2Seq(en_dims=en_dims, 
                  zh_dims=zh_dims,
                  dropout=dropout,
                  en_hidden=en_hidden, 
                  zh_hidden=zh_hidden,
                  atten_vec_size=atten_vec_size,
                  channels=channels,
                  kernel_size=kernel_size,
                  entext_len=60)
    
    pre_trained = torch.load('/data/xuwenshen/ai_challenge/code/bengio_cnn_words/models/ssprob-0.778805-loss-5.048128-score-0.371117-steps-100000-model.pkl') 
    net.load_state_dict(pre_trained)
    print(net)
    
    
    epoch = 10000
    lr = 0.001

    
    test(test_loader=test_loader,
          transform=transform,
          net=net,
          batch_size=batch_size)
    
print ('done')
