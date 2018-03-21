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
import math
from torch.nn import utils

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from Utils import Utils
from net import Seq2Seq
from hyperboard import Agent
from BLEUScore import BLEUScore
from Folder import Folder
from Transform import Transform

bleuscore = BLEUScore()
model_dir = '/data/xuwenshen/ai_challenge/code/bengio/models/'

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
        zhgtruth = torch.Tensor([[transform.zh_go_id for j in range(101)] for i in range(entext.size(0))])

        logits, predic, rnn_enc = net(entext, zhgtruth, enlen, 0, False)

        all_pre.extend(predic)
        all_en.extend(entext.tolist())
        
        del logits, predic

        
    fout = open('/data/xuwenshen/ai_challenge/code/bengio/pred.txt', 'w')
    for i in range(len(all_pre)):
        all_pre[i] = transform.clip(all_pre[i], language='zh')
        all_en[i] = transform.clip(all_en[i], language='en')
        
        en_text = transform.i2t(all_en[i], language='en')
        zh_gen = transform.i2t(all_pre[i], language='zh')
        fout.write(zh_gen + '\n')
            
    fout.close()
    

    
    

if __name__ == '__main__':
    
    
    batch_size = 48
    nb_samples = 8000
    path = '/data/xuwenshen/ai_challenge/data/valid/valid/valid.h5py'
    
    test_folder = Folder(filepath=path,
                         is_test=True,
                         nb_samples=nb_samples)
    test_loader = DataLoader(test_folder,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=False)
    
    
    en_voc_path = '/data/xuwenshen/ai_challenge/data/train/train/en_voc.json'
    zh_voc_path = '/data/xuwenshen/ai_challenge/data/train/train/zh_voc.json'
    transform = Transform(en_voc_path=en_voc_path,
                         zh_voc_path=zh_voc_path)
    
    
    en_dims = 800
    en_voc = 50004
    zh_dims = 800
    zh_voc = 4004
    en_hidden = 800
    zh_hidden = 1000
    atten_vec_size = 1200
    
    net = Seq2Seq(en_dims=en_dims, 
                  en_voc=en_voc,
                  zh_dims=zh_dims,
                  zh_voc=zh_voc,
                  dropout=1,
                  en_hidden=en_hidden, 
                  zh_hidden=zh_hidden,
                  atten_vec_size=atten_vec_size,
                  entext_len=60)
    
    pre_trained = torch.load('/data/xuwenshen/ai_challenge/code/bengio/models/ssprob-0.666313-loss-5.194733-score-0.339406-steps-41200-model.pkl') 
    net.load_state_dict(pre_trained)
    print (net)
   
    
    test(test_loader=test_loader,
          transform=transform,
          net=net,
          batch_size=batch_size)
    
