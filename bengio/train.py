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


def train(lr, net, epoch, train_loader, valid_loader, transform, hyperparameters, batch_size):

    # register hypercurve
    agent = Agent(port=5000)
    hyperparameters['criteria'] = 'train loss'
    train_loss = agent.register(hyperparameters, 'loss')
    
    hyperparameters['criteria'] = 'valid loss'
    valid_loss = agent.register(hyperparameters, 'loss')
    
    hyperparameters['criteria'] = 'valid bleu'
    valid_bleu = agent.register(hyperparameters, 'bleu')
    
    hyperparameters['criteria'] = 'train bleu'
    train_bleu = agent.register(hyperparameters, 'bleu')
    
    hyperparameters['criteria'] = 'scheduled sampling probability'
    hyper_ssprob = agent.register(hyperparameters, 'probability')
    
    if torch.cuda.is_available():
        net.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
    net.train()
    
    best_score = -1
    global_steps = 0
    best_valid_loss  = 10000
    for iepoch in range(epoch):
        
        new_epoch = False
        batchid = 0
        for (_, data) in enumerate(train_loader, 0):

            entext = data['entext']
            enlen = data['enlen']
            zhlabel = data['zhlabel']
            zhgtruth = data['zhgtruth']
            zhlen = data['zhlen']
            
            ssprob = max(math.exp(-(global_steps)/200000-0.2), 0.5)

            print ('scheduled sampling pro: ', ssprob)
            logits, predic, rnn_enc = net(entext, zhgtruth, enlen, ssprob, True)
            loss = net.get_loss(logits, zhlabel)
            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm(net.parameters(), 5)
            optimizer.step()
            
            batchid += 1
            global_steps += 1
            
            print (global_steps, iepoch, batchid, max(enlen), sum(loss.data.cpu().numpy()))
            agent.append(train_loss, global_steps, sum(loss.data.cpu().numpy()))
            agent.append(hyper_ssprob, global_steps, ssprob)
            
            if batchid % 50 == 0:
                net.eval()
                logits, predic, rnn_enc = net(entext, zhgtruth, enlen, ssprob, True)
                
                tmppre = [0 for i in range(len(entext))]
                tmplabel = [0 for i in range(len(entext))]
                for i in range(len(zhlabel)):
                    tmppre[i] = transform.clip(predic[i], language='zh')
                    tmplabel[i] = zhlabel[i][: zhlen[i]]
                
                tmpscore = bleuscore.score(tmppre, tmplabel)   
                
                for i in range(5):
                    ans_ = transform.i2t(tmplabel[i], language = 'zh')
                    pre_ = transform.i2t(tmppre[i], language = 'zh')
                    print (ans_)
                    print (pre_)
                    print ('-------------------\n')
                
                del logits, predic
                agent.append(train_bleu, global_steps, tmpscore)
                net.train()
            
            if batchid % 400 == 0:
                print ('\n------------------------\n')
                net.eval()
                all_pre = []
                all_lable = []
                all_len = []
                all_loss = 0
                bats = 0
                for (_, data) in enumerate(valid_loader, 0):
                    
                    entext = data['entext']
                    enlen = data['enlen']
                    zhlabel = data['zhlabel']
                    zhgtruth = data['zhgtruth']
                    zhlen = data['zhlen']

                    logits, predic, rnn_enc = net(entext, zhgtruth, enlen, 0, False)
                    loss = net.get_loss(logits, zhlabel)
                    
                    all_pre.extend(predic)
                    all_lable.extend(zhlabel)
                    all_len.extend(zhlen)
                    all_loss += sum(loss.data.cpu().numpy())
                    
                    del loss, logits, predic
                    bats += 1
                
                for i in range(len(all_pre)):
                    all_pre[i] = transform.clip(all_pre[i], language='zh')
                    all_lable[i] = all_lable[i][: all_len[i]]

                score = bleuscore.score(all_pre, all_lable)
            
                for i in range(0, 600, 6):
                    ans_ = transform.i2t(all_lable[i], language = 'zh')
                    pre_ = transform.i2t(all_pre[i], language = 'zh')
                    print (ans_)
                    print (pre_)
                    print ('-------------------\n')
        
                all_loss /= bats
                print (global_steps, iepoch, batchid, all_loss, score, '\n********************\n')
                agent.append(valid_loss, global_steps, all_loss)
                agent.append(valid_bleu, global_steps, score)
                
                if best_score < score or best_valid_loss > all_loss:
                    
                    best_valid_loss = all_loss
                    bestscore = score
                    torch.save(net.state_dict(), model_dir + "ssprob-{:3f}-loss-{:3f}-score-{:3f}-steps-{:d}-model.pkl".format(ssprob, all_loss, score, global_steps))
                del all_lable, all_len, all_loss, all_pre
                net.train()

    
    

if __name__ == '__main__':
    
    
    batch_size = 48
    nb_samples = 9893707
    path = '/data/xuwenshen/ai_challenge/data/train/train/train.h5py'
    
    train_folder = Folder(filepath=path,
                          is_test=False,
                          nb_samples=nb_samples)
    train_loader = DataLoader(train_folder,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=True)
    
    
    path = '/data/xuwenshen/ai_challenge/data/valid/valid/valid.h5py'
    nb_samples = 8000
    valid_folder = Folder(filepath=path,
                          is_test=False,
                          nb_samples=nb_samples)
    valid_loader = DataLoader(valid_folder,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=True)
    
    en_voc_path = '/data/xuwenshen/ai_challenge/data/train/train/en_voc.json'
    zh_voc_path = '/data/xuwenshen/ai_challenge/data/train/train/zh_voc.json'
    transform = Transform(en_voc_path=en_voc_path,
                         zh_voc_path=zh_voc_path)
    
    
    en_dims = 800
    en_voc = 50004
    zh_dims = 800
    zh_voc = 4004 
    dropout = 0.5
    en_hidden = 800
    zh_hidden = 1000
    atten_vec_size = 1200
    
    net = Seq2Seq(en_dims=en_dims, 
                  en_voc=en_voc,
                  zh_dims=zh_dims,
                  zh_voc=zh_voc,
                  dropout=dropout,
                  en_hidden=en_hidden, 
                  zh_hidden=zh_hidden,
                  atten_vec_size=atten_vec_size,
                  entext_len=60)
    
    pre_trained = torch.load('./models/saved_models/ssprob-1.000000-loss-7.136847-score-0.335522-steps-128750model.pkl') 
    net.load_state_dict(pre_trained)
    print (net)
   
    
    epoch = 10000
    lr = 0.001

    hyperparameters = {
        'learning rate': lr,
        'batch size': batch_size,
        'criteria': '',
        'dropout': dropout,
        'atten_vec_size':atten_vec_size,
        'en_hidden': en_hidden,
        'zh_hidden': zh_hidden,
        'en_embedding': en_dims,
        'zh_embedding': zh_dims
    }
    
    train(lr=lr,
          train_loader=train_loader,
          valid_loader=valid_loader,
          transform=transform,
          net=net,
          batch_size=batch_size,
          hyperparameters=hyperparameters,
          epoch=epoch)
    
