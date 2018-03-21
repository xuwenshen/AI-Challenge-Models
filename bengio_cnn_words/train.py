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

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from net import Seq2Seq
from hyperboard import Agent
from BLEUScore import BLEUScore
from Folder import Folder
from Transform import Transform

bleuscore = BLEUScore()
model_dir = '/data/xuwenshen/ai_challenge/code/bengio_cnn_words/models/'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


def train(lr, net, epoch, train_loader, valid_loader, transform, hyperparameters, batch_size):

    # register hypercurve
    agent = Agent(port=5004)
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
            enstr = data['enstr']
            zhstr = data['zhstr']
            
            ssprob = max(math.exp(-global_steps/200000-0.23), 0.5)

            print ('scheduled sampling pro: ', ssprob)
            logits, predic = net(entext, zhgtruth, enlen, ssprob, True)
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
                logits, predic = net(entext, zhgtruth, enlen, ssprob, True)
                
                tmppre = [0 for i in range(len(entext))]
                tmplabel = [0 for i in range(len(entext))]
                for i in range(len(zhlabel)):
                    tmppre[i] = transform.clip(predic[i], language='zh')
                    tmppre[i] = transform.i2t(tmppre[i], language = 'zh')
                    tmppre[i] = re.sub(r'nuk#', '', tmppre[i])
                
                tmpscore = bleuscore.score(tmppre, zhstr)   
                
                for i in range(5):
                    print (tmppre[i])
                    print (zhstr[i])
                    print ('-------------------\n')
                
                del logits, predic
                agent.append(train_bleu, global_steps, tmpscore)
                net.train()
            
            if batchid % 100 == 0:
                print ('\n------------------------\n')
                net.eval()
                
                all_pre = []
                all_label = []
                all_len = []
                all_loss = 0
                bats = 0
                
                for (_, data) in enumerate(valid_loader, 0):
                    
                    entext = data['entext']
                    enlen = data['enlen']
                    zhlabel = data['zhlabel']
                    zhgtruth = data['zhgtruth']
                    zhlen = data['zhlen']
                    enstr = data['enstr']
                    zhstr = data['zhstr']
                    
                    logits, predic = net(entext, zhgtruth, enlen, 0, False)
                    loss = net.get_loss(logits, zhlabel)
                    
                    all_pre.extend(predic)
                    all_label.extend(zhstr)
                    all_len.extend(zhlen)
                    all_loss += sum(loss.data.cpu().numpy())
                    
                    del loss, logits, predic
                    bats += 1
                
                for i in range(len(all_pre)):
                    all_pre[i] = transform.clip(all_pre[i], language='zh')
                    all_pre[i] = transform.i2t(all_pre[i], language = 'zh')
                    all_pre[i] = re.sub(r'nuk#', '', all_pre[i])

                score = bleuscore.score(all_pre, all_label)
            
                for i in range(0, 600, 6):
                    print (all_pre[i])
                    print (all_label[i])
                    print ('-------------------\n')
        
                all_loss /= bats
                print (global_steps, iepoch, batchid, all_loss, score, '\n********************\n')
                agent.append(valid_loss, global_steps, all_loss)
                agent.append(valid_bleu, global_steps, score)
                
                if best_score < score or best_valid_loss > all_loss:
                    
                    best_valid_loss = all_loss
                    bestscore = score
                    torch.save(net.state_dict(), model_dir + "ssprob-{:3f}-loss-{:3f}-score-{:3f}-steps-{:d}-model.pkl".format(ssprob, all_loss, score, global_steps))
                del all_label, all_len, all_loss, all_pre
                net.train()

    

if __name__ == '__main__':
    
    
    batch_size = 32
    nb_samples = 9886000
    path = '/data/xuwenshen/ai_challenge/data/train/train_words/train.h5py'
    
    train_folder = Folder(filepath=path,
                          is_test=False,
                          nb_samples=nb_samples)
    train_loader = DataLoader(train_folder,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=True)
    
    
    path = '/data/xuwenshen/ai_challenge/data/valid/valid_words/valid.h5py'
    nb_samples = 2000
    valid_folder = Folder(filepath=path,
                          is_test=False,
                          nb_samples=nb_samples)
    valid_loader = DataLoader(valid_folder,
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
    
    pre_trained = torch.load('./models/ssprob-0.777249-loss-5.034320-score-0.366200-steps-100400-model.pkl') 
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
        'zh_embedding': zh_dims,
        'channels': channels,
        'kernel_size': kernel_size
    }
    
    train(lr=lr,
          train_loader=train_loader,
          valid_loader=valid_loader,
          transform=transform,
          net=net,
          batch_size=batch_size,
          hyperparameters=hyperparameters,
          epoch=epoch)
    