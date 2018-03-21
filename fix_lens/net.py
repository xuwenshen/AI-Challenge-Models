import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

import numpy as np 
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import json
import random
import torch.nn.utils.rnn as rnn_utils

from Transform import Transform

from Transform import Transform

transform = Transform(zh_voc_path='/data/xuwenshen/ai_challenge/data/train/train/zh_voc.json',
               en_voc_path='/data/xuwenshen/ai_challenge/data/train/train/en_voc.json')

weight = [1 for i in range(len(transform.zh_voc))]
weight[transform.zh_pad_id] = 0


    
class EncLSTM(nn.Module):
    
    def __init__(self, dropout_p, en_hidden, en_dims, enc_layers):
        
        super(EncLSTM, self).__init__()
        
        self.en_hidden = en_hidden
        
        self.enc_lstm = torch.nn.LSTM(input_size=en_dims,
                                      num_layers=enc_layers,
                                      hidden_size=en_hidden, 
                                      dropout=dropout_p,
                                      bidirectional=True,
                                      batch_first=True)
        
        
    def forward(self, inputs):    
        
        outputs, states = self.enc_lstm(inputs)
        return outputs, states
      

class Dec(nn.Module):
    
    def __init__(self, zh_max_len, zh_hidden, dec_layers, input_dropout_p, dropout_p, beam_size, zh_embedding_size):
        
        super(Dec, self).__init__()
        
        self.dec_rnn = DecoderRNN(vocab_size = len(transform.zh_voc),
                                  max_len = zh_max_len,
                                  embedding_size = zh_embedding_size,
                                  hidden_size = zh_hidden,
                                  sos_id = transform.zh_go_id,
                                  eos_id = transform.zh_eos_id, 
                                  n_layers = dec_layers,
                                  rnn_cell='lstm',
                                  bidirectional=True,
                                  input_dropout_p = input_dropout_p,
                                  dropout_p=dropout_p,
                                  use_attention=True)
        
        self.beam_dec = TopKDecoder(self.dec_rnn, beam_size)
        
    def forward(self, gtruths, encoder_hidden, encoder_outputs, teacher_forcing_ratio, is_train):
        
        
        if is_train:
            if teacher_forcing_ratio > 0:
                gtruths = Variable(gtruths.long()).cuda()
            decoder_outputs, decoder_hidden, ret_dict = self.dec_rnn(inputs = gtruths,
                                                                     encoder_hidden = encoder_hidden,
                                                                     encoder_outputs = encoder_outputs,
                                                                     teacher_forcing_ratio = teacher_forcing_ratio)
        else:
            decoder_outputs, decoder_hidden, ret_dict = self.beam_dec(inputs = gtruths,
                                                                      encoder_hidden = encoder_hidden,
                                                                      encoder_outputs = encoder_outputs,
                                                                      teacher_forcing_ratio = teacher_forcing_ratio)
            
        return decoder_outputs, decoder_hidden, ret_dict
    
    
class Seq2Seq(nn.Module):
    
    def __init__(self, en_dims, zh_dims, input_dropout_p, dropout_p, en_hidden, zh_hidden, enc_layers, dec_layers, en_max_len, zh_max_len, beam_size):
        
        super(Seq2Seq, self).__init__()
        
        self.weight = torch.Tensor(weight)
        self.input_dropout = nn.Dropout(input_dropout_p)
        
        self.en_embedding = torch.nn.Embedding(num_embeddings=len(transform.en_voc), embedding_dim=en_dims)
        
        self.cost_func = nn.NLLLoss(weight=self.weight)
        
        self.rnn_net = EncLSTM(dropout_p=dropout_p, en_dims=en_dims, en_hidden=en_hidden, enc_layers=enc_layers)
        
        self.dec_net = Dec(zh_max_len=zh_max_len, 
                            zh_hidden=zh_hidden,
                            zh_embedding_size=zh_dims,
                            dec_layers=dec_layers,
                            input_dropout_p=input_dropout_p,
                            dropout_p=dropout_p, 
                            beam_size = beam_size)
        
        
    def forward(self, inputs, gtruths, is_train, teacher_forcing_ratio):
        
        inputs = Variable(inputs).long().cuda()
                
        inputs = self.en_embedding(inputs)
        inputs = self.input_dropout(inputs)

        rnn_enc, enc_hidden = self.rnn_net(inputs)
        
        decoder_outputs, decoder_hidden, ret_dict =  self.dec_net(gtruths=gtruths,
                                                                  encoder_hidden=enc_hidden,
                                                                  encoder_outputs=rnn_enc,
                                                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                                                  is_train=is_train)
        
        return decoder_outputs, ret_dict


    def get_loss(self, logits, labels):
        
        labels = Variable(labels).long().cuda()
        labels = labels.transpose(0, 1)
        
        for i in range(len(logits)):
            logits[i] = logits[i].contiguous().view(1, logits[i].size(0), logits[i].size(1))
        logits = torch.cat(logits)
        
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss
