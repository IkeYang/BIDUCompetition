#Author:ike yang
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import copy
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class Embedding(nn.Module):
    def __init__(self, p,embedingDim,tuo=1,wt=33,wl=50):
        super().__init__()
        self.relu=nn.ReLU()
        self.sftm=nn.Softmax(dim=-1)
        self.tuo=tuo
        self.thres=1/p/5/wt
        self.weight = Parameter(
            torch.ones(int(p*wt*wl)))

        self.FC=nn.Linear(p*wt,embedingDim)
    def forward(self, x):
        bs,wl,wt,p=x.shape
        x=x.view(bs,-1)
        w = self.sftm(self.weight/self.tuo)
        x=x * w
        x = x.view(bs, wl, -1)
        x =self.FC(x).squeeze(dim=-1)
        # x=rearrange(x,'b c p-> b p c')### p=wt*numberWT,c=wl
        return x

class network(nn.Module):
    def __init__(self, wt, p, dropout=0,windowLength=6*24*5,embedding=False):
        super(network, self).__init__()
        embedingDim=128


        self.lstm1 = torch.nn.LSTM(embedingDim, embedingDim, bidirectional =True,  dropout=dropout,batch_first=True)


        self.embedding_layer1 =Embedding(p,embedingDim=embedingDim,tuo=1,wt=wt,wl=windowLength)

        if embedding:
            self.mlp1 = torch.nn.Linear(embedingDim+wt*p, embedingDim)
        else:
            self.mlp1 = torch.nn.Linear(wt * p, embedingDim)
        self.mlp_bn1 = torch.nn.BatchNorm1d(embedingDim)


        self.mlp2 = torch.nn.Linear(6*embedingDim, 2*embedingDim)
        self.mlp_bn2 = torch.nn.BatchNorm1d(2*embedingDim)

        self.lstm_out1 = torch.nn.LSTM(2*embedingDim, 2*embedingDim, bidirectional =True, dropout=dropout,batch_first=True)
        self.lstm_out2 = torch.nn.LSTM(4*embedingDim, embedingDim, bidirectional =True,  dropout=dropout,batch_first=True)
        self.lstm_out3 = torch.nn.LSTM(2*embedingDim, int(embedingDim/2), bidirectional =True,  dropout=dropout,batch_first=True)
        self.lstm_out4 = torch.nn.LSTM(int(embedingDim/2)*2, int(embedingDim/2)*2, bidirectional =True,  dropout=dropout,batch_first=True)

        self.output = torch.nn.Linear(int(embedingDim/2)*2*2, 2*6*24*2 )

        self.sigmoid = torch.nn.Sigmoid()
        self.embedding = embedding

        # 网络的前向计算函数

    def forward(self, input1):
        #input1 bs,wl,wt,p
        bs, wl, wt, p=input1.shape
        if self.embedding:
            embedded1 = self.embedding_layer1(input1)#bs , wl, embedingDim
            input1=input1.view(bs,wl,-1)

            x = torch.cat([
                embedded1,
                input1
            ], dim=-1)  # bs,wl, wt*p+embedingDim
        else:
            x = input1.view(bs, wl, -1)

        x = self.mlp1(x)
        x = rearrange(x, 'b c p-> b p c')
        x = self.mlp_bn1(x)
        x = rearrange(x, 'b c p-> b p c')
        x = torch.nn.ReLU()(x)

        x_lstm_out, (hidden, _) = self.lstm1(x)
        hidden=rearrange(hidden, 'd b p-> b d p').reshape(bs,-1)
        x = torch.cat([
            hidden,
            torch.max(x_lstm_out, dim=1)[0],
            torch.mean(x_lstm_out, dim=1)
        ], dim=-1)#bs, embedingDim+embedingDim+2*embedingDim+2*embedingDim

        x = self.mlp2(x)

        x = self.mlp_bn2(x)

        x = torch.nn.ReLU()(x)

        # decoder

        x = torch.stack([x] * 20, dim=1)

        x = self.lstm_out1(x)[0]
        x = self.lstm_out2(x)[0]
        x = self.lstm_out3(x)[0]
        x = self.lstm_out4(x)[0][:,-1,:]#bs, wl,  embedingDim*2

        output = self.output(x)#bs,6*24*2*2
        # output = self.sigmoid(x) * 2 - 1
        # output = torch.cast(output, dtype='float32')

        return output

if __name__=='__main__':
    wt=100
    p=9
    rnn = network(wt, p, dropout=0,windowLength=6*24*5,embedding=True)
    input = torch.randn(16, 6*24*5, wt,p)
    out = rnn(input)














































