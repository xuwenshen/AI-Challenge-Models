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
import torch.nn.utils.rnn as rnn_utils

from Transform import Transform

transform = Transform(zh_voc_path='/data/xuwenshen/ai_challenge/data/train/train_words/zh_voc.json',
               en_voc_path='/data/xuwenshen/ai_challenge/data/train/train_words/en_voc.json')

en_voc_size = len(transform.en_voc)
zh_voc_size = len(transform.zh_voc)

weight = [1 for i in range(zh_voc_size)]
weight[transform.zh_pad_id] = 0

class Enc_cnn(nn.Module):
    
    def __init__(self, dropout, channels, kernel_size, en_dims, entext_len):
        
        super(Enc_cnn, self).__init__() 
        
        self.conv = nn.Conv1d(in_channels= en_dims, out_channels= channels, kernel_size= kernel_size)
        
        self.pool = nn.MaxPool1d(entext_len-kernel_size+1)
        
        self.batchNorm = nn.BatchNorm1d(channels)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        
    
    def forward(self, inputs):
        
        inputs = torch.transpose(inputs, 1, 2)
        
        layer = self.batchNorm(self.conv(inputs))
        layer = self.dropout(self.relu(layer))
        conv_enc = self.pool(layer)
        
        conv_enc = conv_enc.view(conv_enc.size(0), conv_enc.size(1))

        return conv_enc
        



class Enc_lstm(nn.Module):
    
    def __init__(self, dropout, en_hidden, en_dims):
        
        super(Enc_lstm, self).__init__()
        
        self.en_hidden = en_hidden
        
        self.enc_lstm = torch.nn.LSTM(input_size=en_dims,
                                         hidden_size=en_hidden, 
                                         dropout=dropout,
                                         bidirectional=True,
                                         batch_first=False)
        
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, inputs, inputs_len):

        inputs = torch.transpose(inputs, 0, 1)

        #实现Variable lengths输入RNN
        packed = rnn_utils.pack_padded_sequence(input=inputs, lengths=list(inputs_len), batch_first =False)
        packed_out, _ = self.enc_lstm(packed)
        unpacked_, _ = rnn_utils.pad_packed_sequence(packed_out)
        
        return unpacked_.transpose(0, 1)
        
        

class Dec(nn.Module):
    
    def __init__(self, zh_dims, dropout, zh_hidden, rnn_enc_size, cnn_enc_size, atten_vec_size):
        
        super(Dec, self).__init__()

        self.zh_hidden = zh_hidden
        self.zh_embedding = torch.nn.Embedding(num_embeddings=zh_voc_size, embedding_dim=zh_dims)
        
        self.dec_lstm_cell = torch.nn.LSTMCell(input_size=rnn_enc_size+zh_dims+cnn_enc_size,
                                  hidden_size=zh_hidden, 
                                  bias=True)
        
        self.atten_ws = nn.Linear(in_features=zh_hidden, out_features=atten_vec_size)
        self.atten_uh = nn.Linear(in_features = rnn_enc_size, out_features = atten_vec_size)
        self.atten_v = nn.Linear(in_features = atten_vec_size, out_features = 1)
        
        self.fc = nn.Linear(in_features=zh_hidden, out_features=zh_voc_size)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
        
    def forward(self, rnn_enc, cnn_enc, gtruths, ssprob, is_train):
        
        gtruths = Variable(gtruths).long().cuda()
        gtruths = self.zh_embedding(gtruths)
        
        hx = Variable(torch.zeros(rnn_enc.size(0), self.zh_hidden)).cuda()
        cx = Variable(torch.zeros(rnn_enc.size(0), self.zh_hidden)).cuda()
        
        gtruths = torch.transpose(gtruths, 0, 1)
        
        logits = [0 for i in range(gtruths.size(0))]
        predic = [0 for i in range(gtruths.size(0))]
        
        Uh = self.atten_uh(rnn_enc.transpose(0, 1))
        
        for i in range(gtruths.size(0)):
            
            ##attention
            ws = self.atten_ws(hx)
            ws = ws.expand(Uh.size(0), ws.size(0), ws.size(1))
            sum_ = Uh+ws
            sum_ = self.tanh(sum_)
            score = self.atten_v(sum_)
            score = score.view(score.size(0), score.size(1))
            score = torch.transpose(score, 0, 1)
            score = self.softmax(score)
            score = score.view(score.size(0), 1, score.size(1))
            
            atten_vec = torch.bmm(score, rnn_enc)
            atten_vec = atten_vec.view(atten_vec.size(0), atten_vec.size(-1))
                
            ##attention end
            
            in_vec = torch.cat([atten_vec, cnn_enc], 1)
            
            if is_train:
                
                if random.random() > ssprob and i > 0:
                    prev = self.zh_embedding(predic[i-1])
                    prev = prev.view(prev.size(1), prev.size(2))
                    inp = torch.cat([in_vec, prev], -1)
                    
                else:
                    inp = torch.cat([in_vec, gtruths[i]], -1)
                
                    
            else:
                if i == 0:
                    inp = torch.cat([in_vec, gtruths[0]], -1)
                else:
                    prev = self.zh_embedding(predic[i-1])
                    prev = prev.view(prev.size(1), prev.size(2))
                    inp = torch.cat([in_vec, prev], -1)
                
            hx, cx = self.dec_lstm_cell(inp, (hx, cx))
            logits[i] = self.fc(hx)
            
            _, predic[i] = torch.max(logits[i], 1)
            logits[i] = logits[i].view(1, logits[i].size(0), logits[i].size(1))
            
            predic[i] = predic[i].view(1, predic[i].size(0))
        
        predic = torch.cat(predic, 0)
        predic = torch.transpose(predic, 0, 1)
        
        
        return torch.cat(logits), predic.data.cpu().numpy()
    
    
    
class Seq2Seq(nn.Module):
    
    def __init__(self, en_dims, zh_dims, dropout, en_hidden, zh_hidden, entext_len, atten_vec_size, channels, kernel_size):
        
        super(Seq2Seq, self).__init__()
        
        self.weight = torch.Tensor(weight)
        
        self.en_embedding = torch.nn.Embedding(num_embeddings=en_voc_size, embedding_dim=en_dims)
        
        self.cost_func = nn.CrossEntropyLoss(weight=self.weight)
        
        self.cnn_net = Enc_cnn(dropout=dropout, channels=channels, kernel_size=kernel_size, en_dims=en_dims, entext_len=entext_len)
        self.rnn_net = Enc_lstm(dropout=dropout, en_dims=en_dims, en_hidden=en_hidden)
        
        #保存中间enc值
        self.cnn_enc = None
        self.rnn_enc = None
        
        self.dec_net = Dec(zh_dims=zh_dims, dropout=dropout, zh_hidden=zh_hidden, rnn_enc_size=en_hidden*2,cnn_enc_size=channels, atten_vec_size=atten_vec_size)

 
    def order(self, inputs, inputs_len):
        
        inputs_len, sort_ids = torch.sort(inputs_len, 0, descending=True)
        inputs = inputs.index_select(0, Variable(sort_ids).cuda())
        
        _, true_order_ids = torch.sort(sort_ids, 0, descending=False)
        
        return inputs, inputs_len, true_order_ids
        
    def forward(self, inputs, gtruths, inputs_len, ssprob, is_train):
        
        inputs = Variable(inputs).long().cuda()
        
        # change order according to lens
        inputs, sort_len, true_order_ids = self.order(inputs, inputs_len)
        
        inputs = self.en_embedding(inputs)

        rnn_enc = self.rnn_net(inputs, sort_len)
        cnn_enc = self.cnn_net(inputs)
        
        self.rnn_enc = rnn_enc.index_select(0, Variable(true_order_ids).cuda())
        self.cnn_enc = cnn_enc.index_select(0, Variable(true_order_ids).cuda())
        
        logits, predic = self.dec_net(self.rnn_enc, self.cnn_enc, gtruths, ssprob, is_train)
        
        return logits, predic

    def get_loss(self, logits, labels):
        
        labels = Variable(labels).long().cuda()
        labels = labels.transpose(0, 1)
        
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss
    
    def get_cnn_enc(self):
        return self.cnn_enc
    
    def get_rnn_enc(self):
        return self.rnn_enc
    
    def get_cnn_enc_net(self):
        return self.cnn_net
    
    def get_rnn_enc_net(self):
        return self.rnn_net
    
    def get_dec_net(self):
        return self.dec_net