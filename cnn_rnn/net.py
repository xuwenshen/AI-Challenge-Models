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

transform = Transform(zh_voc_path='/data/xuwenshen/ai_challenge/data/train/train/zh_voc.json',
               en_voc_path='/data/xuwenshen/ai_challenge/data/train/train/en_voc.json')

weight = [1 for i in range(4004)]
weight[transform.zh_pad_id] = 0


class Enc_cnn(nn.Module):
    
    def __init__(self, dropout, channels, kernel_size, en_dims, entext_len):
        
        super(Enc_cnn, self).__init__() 
        
        self.conv1 = nn.Conv1d(in_channels= en_dims, out_channels= channels, kernel_size= kernel_size)
        self.conv2 = nn.Conv1d(in_channels= channels, out_channels= channels, kernel_size= kernel_size)
        self.conv3 = nn.Conv1d(in_channels= channels, out_channels= channels, kernel_size= kernel_size)
        self.conv4 = nn.Conv1d(in_channels= channels, out_channels= channels, kernel_size= kernel_size)
        
        self.pool1 = nn.MaxPool1d(entext_len-kernel_size+1)
        self.pool2 = nn.MaxPool1d(entext_len-2*kernel_size+2)
        self.pool3 = nn.MaxPool1d(entext_len-3*kernel_size+3)
        self.pool4 = nn.MaxPool1d(entext_len-4*kernel_size+4)

        self.batchNorm1 = nn.BatchNorm1d(channels)
        self.batchNorm2 = nn.BatchNorm1d(channels)
        self.batchNorm3 = nn.BatchNorm1d(channels)
        self.batchNorm4 = nn.BatchNorm1d(channels)
        
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        
    
    def forward(self, inputs):
        
        inputs = torch.transpose(inputs, 1, 2)
        
        layer1 = self.batchNorm1(self.conv1(inputs))
        layer1 = self.dropout(self.relu(layer1))
        pool1 = self.pool1(layer1)
        
        
        layer2 = self.batchNorm2(self.conv2(layer1))
        layer2 = self.dropout(self.relu(layer2))
        pool2 = self.pool2(layer2)
        
        layer3 = self.batchNorm3(self.conv3(layer2))
        layer3 = self.dropout(self.relu(layer3))
        pool3 = self.pool3(layer3)
        
        layer4 = self.batchNorm4(self.conv4(layer3))
        layer4 = self.dropout(self.relu(layer4))
        pool4 = self.pool4(layer4)
        
        conv_enc = torch.cat([pool1, pool2, pool3, pool4], 1)
        conv_enc = conv_enc.view(conv_enc.size(0), conv_enc.size(1))

     
        return conv_enc
        

class Enc_lstm(nn.Module):
    
    def __init__(self, dropout, en_hidden, en_dims, enc_conv_len):
        
        super(Enc_lstm, self).__init__()
        
        self.en_hidden = en_hidden
        
        self.enc_fw_lstm = torch.nn.LSTM(input_size=en_dims+enc_conv_len,
                                         hidden_size=en_hidden, 
                                         dropout=dropout,
                                         bidirectional=True,
                                         batch_first=False)
        
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, inputs, conv_enc, inputs_len):
        
        inputs = torch.transpose(inputs, 0, 1)
        conv_enc = conv_enc.expand(inputs.size(0), conv_enc.size(0), conv_enc.size(1))
        
        inputs = torch.cat([inputs, conv_enc], -1)
        
        enl = list(inputs_len)
        
        #实现Variable lengths输入RNN
        packed = rnn_utils.pack_padded_sequence(input=inputs, lengths=enl, batch_first =False)
        packed_out, _ = self.enc_fw_lstm(packed)
        unpacked_, _ = rnn_utils.pad_packed_sequence(packed_out)
        
        lstm_enc = [0 for i in range(len(enl))]
        for i in range(len(enl)):
            lstm_enc[i] = torch.cat([unpacked_[enl[i]-1, i, :self.en_hidden], unpacked_[0, i, self.en_hidden:]])
            lstm_enc[i] = lstm_enc[i].view(1, lstm_enc[i].size(0))
        
        lstm_enc = torch.cat(lstm_enc)
        
        
        return lstm_enc
        
        

class Dec(nn.Module):
    
    def __init__(self, zh_dims, zh_voc, dropout, zh_hidden, enc_hidden):
        
        super(Dec, self).__init__()

        self.zh_hidden = zh_hidden
        self.zh_embedding = torch.nn.Embedding(num_embeddings=zh_voc, embedding_dim=zh_dims)
        
        
        self.dec_lstm_cell = torch.nn.LSTMCell(input_size=enc_hidden+zh_dims,
                                  hidden_size=zh_hidden, 
                                  bias=True)
        
        self.fc = nn.Linear(in_features=self.zh_hidden, out_features=zh_voc)
        self.softmax = nn.Softmax()
        
        
    def forward(self, encoder, gtruths, ssprob, is_train):
        
        gtruths = Variable(gtruths).long().cuda()
        gtruths = self.zh_embedding(gtruths)
        
        hx = Variable(torch.zeros(encoder.size(0), self.zh_hidden)).cuda()
        cx = Variable(torch.zeros(encoder.size(0), self.zh_hidden)).cuda()
        
        gtruths = torch.transpose(gtruths, 0, 1)
        
        logits = [0 for i in range(gtruths.size(0))]
        predic = [0 for i in range(gtruths.size(0))]
        
        for i in range(gtruths.size(0)):
            
            if is_train:
                
                if random.random() > ssprob and i > 0:
                    prev = self.zh_embedding(predic[i-1])
                    prev = prev.view(prev.size(1), prev.size(2))
                    inp = torch.cat([encoder, prev], -1)
                    
                else:
                    inp = torch.cat([encoder, gtruths[i]], -1)
             
                
                    
            else:
                if i == 0:
                    inp = torch.cat([encoder, gtruths[0]], -1)
                else:
                    prev = self.zh_embedding(predic[i-1])
                    prev = prev.view(prev.size(1), prev.size(2))
                    inp = torch.cat([encoder, prev], -1)
        
            hx, cx = self.dec_lstm_cell(inp, (hx, cx))
            logits[i] = self.fc(hx)

            _, predic[i] = torch.max(logits[i], 1)
            logits[i] = logits[i].view(1, logits[i].size(0), logits[i].size(1))
            
            predic[i] = predic[i].view(1, predic[i].size(0))
        
        predic = torch.cat(predic, 0)
        predic = torch.transpose(predic, 0, 1)
        
        
        return torch.cat(logits), predic.data.cpu().numpy()
    
    
    
class Seq2Seq(nn.Module):
    
    def __init__(self, en_dims, en_voc, zh_dims, zh_voc, dropout, en_hidden, zh_hidden, channels, kernel_size, entext_len):
        
        super(Seq2Seq, self).__init__()
        
        self.enc_conv_len = channels*4
        self.enc_hidden = self.enc_conv_len + en_hidden * 2

        
        self.weight = torch.Tensor(weight)
        
        self.en_embedding = torch.nn.Embedding(num_embeddings=en_voc, embedding_dim=en_dims)
        
        self.cost_func = nn.CrossEntropyLoss(weight=self.weight)
        
        self.conv_net = Enc_cnn(channels=channels, dropout=dropout, en_dims=en_dims, entext_len=entext_len, kernel_size=kernel_size)
        self.rnn_net = Enc_lstm(dropout=dropout, en_dims=en_dims, en_hidden=en_hidden, enc_conv_len=self.enc_conv_len)

        
        self.dec_net = Dec(zh_dims=zh_dims, zh_voc=zh_voc, dropout=dropout, zh_hidden=zh_hidden, enc_hidden=self.enc_hidden)

    def order(self, inputs, inputs_len):
        
        inputs_len, sort_ids = torch.sort(inputs_len, 0, descending=True)
        inputs = inputs.index_select(0, Variable(sort_ids))
        
        _, true_order_ids = torch.sort(sort_ids, 0, descending=False)
        
        return inputs, inputs_len, true_order_ids
    
        
    def forward(self, inputs, gtruths, inputs_len, ssprob, is_train):
        
        
        inputs = Variable(inputs.long()).cuda()
        inputs_len = inputs_len.long().cuda()
        
        # change order according to lens
        sort_inputs, sort_len, true_order_ids = self.order(inputs, inputs_len)
        
        sort_inputs = self.en_embedding(sort_inputs)
        
        conv_enc = self.conv_net(sort_inputs)
        rnn_enc = self.rnn_net(sort_inputs, conv_enc, sort_len)
        encoder = torch.cat([conv_enc, rnn_enc], -1)
        
       
        
        # recover order by true_order_ids
        # 这里需要计算梯度，所以需要用到Variable
        encoder = encoder.index_select(0, Variable(true_order_ids))
        
        
        logits, predic = self.dec_net(encoder, gtruths, ssprob, is_train)
        
        return logits, predic

    def get_loss(self, logits, labels):
        
        labels = Variable(labels).long().cuda()
        labels = torch.transpose(labels, 0, 1)
        
        logits = logits.view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss