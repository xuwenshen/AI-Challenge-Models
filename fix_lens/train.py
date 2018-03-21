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
model_dir = '/data/xuwenshen/ai_challenge/code/fix_lens/models/'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


def train(lr, net, epoch, train_loader, valid_loader, transform, hyperparameters, batch_size):

    # register hypercurve
    agent = Agent(port=5005)
    hyperparameters['criteria'] = 'train loss'
    train_loss = agent.register(hyperparameters, 'loss')
    
    hyperparameters['criteria'] = 'valid loss'
    valid_loss = agent.register(hyperparameters, 'loss')
    
    hyperparameters['criteria'] = 'valid bleu'
    valid_bleu = agent.register(hyperparameters, 'bleu')
    
    hyperparameters['criteria'] = 'train bleu'
    train_bleu = agent.register(hyperparameters, 'bleu')
    
    hyperparameters['criteria'] = 'teacher_forcing_ratio'
    hyper_tfr = agent.register(hyperparameters, 'ratio')
    
    hyperparameters['criteria'] = 'teacher_forcing_loss'
    valid_tf_loss = agent.register(hyperparameters, 'loss')

    if torch.cuda.is_available():
        net.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
    net.train()
    
    best_score = -1
    global_steps = 578800
    best_valid_loss  = 10000
    for iepoch in range(epoch):
        
        batchid = 0
        for (_, tdata) in enumerate(train_loader, 0):

            entext = tdata['entext']
            enlen = tdata['enlen']
            zhlabel = tdata['zhlabel']
            zhgtruth = tdata['zhgtruth']
            zhlen = tdata['zhlen']
            enstr = tdata['enstr']
            zhstr = tdata['zhstr']
            
            teacher_forcing_ratio = 1
            print ('teacher_forcing_ratio: ', teacher_forcing_ratio)
            
            decoder_outputs, ret_dict = net(entext, zhgtruth,True, teacher_forcing_ratio)
            
            
            loss = net.get_loss(decoder_outputs, zhlabel)
            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm(net.parameters(), 5)
            optimizer.step()
            
            batchid += 1
            global_steps += 1
            
            print (global_steps, iepoch, batchid, max(enlen), sum(loss.data.cpu().numpy())) 
            agent.append(train_loss, global_steps, sum(loss.data.cpu().numpy()))
            agent.append(hyper_tfr, global_steps, teacher_forcing_ratio)
            
            if global_steps % 50 == 0:
                net.eval()
                decoder_outputs, ret_dict = net(entext, zhgtruth, True, teacher_forcing_ratio)
                
                length = ret_dict['length']
                prediction = [0 for i in range(len(length))]
                tmppre = [_.squeeze().cpu().data.tolist() for _ in ret_dict['sequence']]
                tmppre = np.array(tmppre).transpose(1, 0)
                
                for i in range(len(tmppre)):
                    
                    prediction[i] = tmppre[i][:length[i]]
                    prediction[i] = transform.i2t(prediction[i], language = 'zh')
                    prediction[i] = re.sub(r'nuk#', '', prediction[i])
                    prediction[i] = re.sub(r'eos#', '', prediction[i])

                tmpscore = bleuscore.score(prediction, zhstr)   
                
                for i in range(5):
                    print (prediction[i])
                    print (zhstr[i])
                    print ('-------------------\n')

                del decoder_outputs, ret_dict
                agent.append(train_bleu, global_steps, tmpscore)
                net.train()
	    
                
            if global_steps % 200 == 0:
                print ('\n------------------------\n')
                net.eval()
                all_pre = []
                all_label = []
                all_loss = 0
                all_en = []
                bats = 0
                teacher_forcing_loss = 0
                for (_, vdata) in enumerate(valid_loader, 0):
                    
                    entext = vdata['entext']
                    enlen = vdata['enlen']
                    zhlabel = vdata['zhlabel']
                    zhgtruth = vdata['zhgtruth']
                    zhlen = vdata['zhlen']
                    enstr = vdata['enstr']
                    zhstr = vdata['zhstr']
                    
                    decoder_outputs, ret_dict = net(entext, None, True, 0)
                    length = ret_dict['length']
                    prediction = [0 for i in range(len(length))]
                    tmppre = [_.squeeze().cpu().data.tolist() for _ in ret_dict['sequence']]
                    tmppre = np.array(tmppre).transpose(1, 0)
                    
                    for i in range(len(tmppre)):
                        prediction[i] = tmppre[i][:length[i]]
                        prediction[i] = transform.i2t(prediction[i], language = 'zh')
                        prediction[i] = re.sub(r'nuk#', '', prediction[i])
                        prediction[i] = re.sub(r'eos#', '', prediction[i])
                    
                    loss = net.get_loss(decoder_outputs, zhlabel)
                    
                    all_pre.extend(prediction)
                    all_label.extend(zhstr)
                    all_en.extend(enstr)
                    all_loss += sum(loss.data.cpu().numpy())
                    
                    del loss, decoder_outputs, ret_dict

                    # teacher forcing loss, to judge if overfit
                    decoder_outputs, _ = net(entext, zhgtruth, True, 1)
                    loss = net.get_loss(decoder_outputs, zhlabel)
                    teacher_forcing_loss += sum(loss.data.cpu().numpy()) 
                    bats += 1
                score = bleuscore.score(all_pre, all_label)
            
                for i in range(0, 400):
                    print (all_en[i])
                    print (all_pre[i])
                    print (all_label[i])
                    print ('-------------------\n')
        
                all_loss /= bats
                teacher_forcing_loss /= bats
                print (global_steps, iepoch, batchid, all_loss, teacher_forcing_loss, score, '\n********************\n')
                agent.append(valid_loss, global_steps, all_loss)
                agent.append(valid_bleu, global_steps, score)
                agent.append(valid_tf_loss, global_steps, teacher_forcing_loss)
                if best_valid_loss > all_loss:
                    
                    best_valid_loss = all_loss
                    #bestscore = score
                    _ = model_dir + "ratio-{:3f}-loss-{:3f}-score-{:3f}-steps-{:d}-model.pkl".format(teacher_forcing_ratio, all_loss, score, global_steps)
                    torch.save(net.state_dict(), _)
                
                elif global_steps % 400 == 0:
                    _ = model_dir + "ratio-{:3f}-loss-{:3f}-score-{:3f}-steps-{:d}-model.pkl".format(teacher_forcing_ratio, all_loss, score, global_steps)
                    torch.save(net.state_dict(), _)

                del all_label, all_loss, all_pre
                net.train()

    
    

if __name__ == '__main__':
    
    
    batch_size = 125
    nb_samples = 9698532
    path = '/data/xuwenshen/ai_challenge/data/train/train/ibm_train-50-60.h5py'
    
    train_folder = Folder(filepath=path,
                          is_test=False,
                          nb_samples=nb_samples)
    train_loader = DataLoader(train_folder,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=True)
    
    
    path = '/data/xuwenshen/ai_challenge/data/valid/valid/ibm_valid-50-60.h5py'
    nb_samples = 640
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
    
    
    en_dims = 712
    en_hidden = 900
    zh_hidden = 1800
    zh_dims = 712
    input_dropout_p = 0.5
    dropout_p = 0.5
    enc_layers = 2
    dec_layers = 2
    en_max_len = 50
    zh_max_len = 61
    beam_size = 5
    
    net = Seq2Seq(en_dims=en_dims,
                  zh_dims=zh_dims,
                  input_dropout_p=input_dropout_p,
                  dropout_p=dropout_p,
                  en_hidden = en_hidden,
                  zh_hidden = zh_hidden,
                  enc_layers = enc_layers,
                  dec_layers = dec_layers,
                  beam_size=beam_size,
                  en_max_len = en_max_len,
                  zh_max_len = zh_max_len)
    
    pre_trained = torch.load('./models/ratio-1.000000-loss-7.047117-score-0.356872-steps-578800-model.pkl') 
    net.load_state_dict(pre_trained)
    
    print (net)
   
    
    epoch = 10000
    lr = 0.001

    hyperparameters = {
        'learning rate': lr,
        'batch size': batch_size,
        'criteria': '',
        'input_dropout_p':input_dropout_p,
        'dropout_p': dropout_p,
        'en_hidden': en_hidden,
        'zh_hidden': zh_hidden,
        'enc_layers':enc_layers,
        'dec_layers':dec_layers,
        'en_dims':en_dims,
        'zh_dims':zh_dims,
        'beam_size':beam_size
    }
    
    train(lr=lr,
          train_loader=train_loader,
          valid_loader=valid_loader,
          transform=transform,
          net=net,
          batch_size=batch_size,
          hyperparameters=hyperparameters,
          epoch=epoch)
    
