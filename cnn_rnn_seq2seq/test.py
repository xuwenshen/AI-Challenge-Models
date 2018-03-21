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
import re
from tqdm import *

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from net import Seq2Seq
from hyperboard import Agent
from BLEUScore import BLEUScore
from Folder import Folder
from Transform import Transform

bleuscore = BLEUScore()
model_dir = '/data/xuwenshen/ai_challenge/code/cnn_rnn_seq2seq/models/'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


def test(net, test_loader, transform, batch_size):

    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    all_pre = []
    
    for (_, data) in enumerate(test_loader, 0):

        entext = data['entext']
        enlen = data['enlen']
        enstr = data['enstr'] # for valid
        
        decoder_outputs, ret_dict = net(entext, None, enlen, False, 0)
        
        print (ret_dict.keys())
        length = [ret_dict['topk_length'][i][0] for i in range(len(ret_dict['topk_length']))] 
        sequences = [ret_dict['topk_sequence'][i] for i in range(len(ret_dict['topk_sequence']))]
        sequences = torch.cat(sequences, -1).cpu()
        #print (sequences.size())
        sequences = sequences.select(1, 0)
        #print (sequences.size())
        #sequences = sequences.squeeze()
        
        prediction = [0 for i in range(len(length))]
        sequences = sequences.data.numpy()
        
        for i in range(len(sequences)):
            prediction[i] = sequences[i][:length[i]]
            prediction[i] = transform.i2t(prediction[i], language = 'zh')
            prediction[i] = re.sub(r'nuk#', '', prediction[i])
            prediction[i] = re.sub(r'eos#', '', prediction[i])
            print (enstr[i])
            print (prediction[i])
            print ('------------------\n')
                    
        all_pre.extend(prediction)
        print (len(all_pre), 'finished....')
        del decoder_outputs, ret_dict, sequences, prediction
        
    print ('writing....')
    fout = open('/data/xuwenshen/ai_challenge/code/cnn_rnn_seq2seq/pred.txt', 'w')
    for i in range(len(all_pre)):
        fout.write(all_pre[i] + '\n')
            
    fout.close()
    
    
    

if __name__ == '__main__':
    
    
    batch_size = 1
    nb_samples = 36
    path = '/data/xuwenshen/ai_challenge/data/valid/valid/ibm_valid-50-60.h5py'
    
    test_folder = Folder(filepath=path,
                         is_test=False,
                         nb_samples=nb_samples)
    test_loader = DataLoader(test_folder,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=False)
    
    
    en_voc_path = '/data/xuwenshen/ai_challenge/data/train/train/en_voc.json'
    zh_voc_path = '/data/xuwenshen/ai_challenge/data/train/train/zh_voc.json'
    transform = Transform(en_voc_path=en_voc_path,
                         zh_voc_path=zh_voc_path)
    
    
    en_dims = 512
    en_hidden = 800
    zh_hidden = 1600
    zh_dims = 512
    input_dropout_p = 0.5
    dropout_p = 0.5
    enc_layers = 2
    dec_layers = 2
    en_max_len = 50
    zh_max_len = 61
    beam_size = 5
    channels = 512
    kernel_size = 3
    
    net = Seq2Seq(en_dims=en_dims,
                  zh_dims=zh_dims,
                  channels = channels,
                  kernel_size = kernel_size,
                  input_dropout_p=input_dropout_p,
                  dropout_p=dropout_p,
                  en_hidden = en_hidden,
                  zh_hidden = zh_hidden,
                  enc_layers = enc_layers,
                  dec_layers = dec_layers,
                  beam_size=beam_size,
                  en_max_len = en_max_len,
                  zh_max_len = zh_max_len)
       
    pre_trained = torch.load('/data/xuwenshen/ai_challenge/code/cnn_rnn_seq2seq/models/ratio-0.992727-loss-6.175089-score-0.343552-steps-73000-model.pkl') 
    net.load_state_dict(pre_trained)
    print (net)
    
    test(test_loader=test_loader,
          transform=transform,
          net=net,
          batch_size=batch_size)
    
