#Author:ike yang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import datetime
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.autograd as autograd
from utlize import SCADADataset
from model import network
import copy



def eval(dataloader, model, device):
    model.eval()
    yNp = np.zeros([1, 288*2])
    yReal = np.zeros([1, 288*2])
    with torch.no_grad():
        c=0
        for i, (x, y) in enumerate(dataloader):
            c += 1
            bs=x.shape[0]
            x = x.to(device)
            y = y.to(device).view(bs,-1)
            y2 = np.copy(y.cpu().detach().numpy())

            ypred = model(x)

            yNp = np.vstack((yNp, ypred .cpu().detach().numpy()))
            yReal = np.vstack((yReal, y2))
    return yNp[1:, :], yReal[1:, :]
def train(deviceNum,nameInput,wtNum,outf,schedulerFunc=ReduceLROnPlateau,server=False,dataPath='E:\YANG Luoxiao\Data\BDWPF\data'):
    WT=134
    P=10


    name='%s__%s'%(nameInput,str(wtNum))
    torch.cuda.set_device(deviceNum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size=64
    lr=2e-4
    weight_decay=0
    Maxepochs=500
    windowLength=6*24*5
    predL=6*24*2
    dropout=0



    scadaTrainDataset = SCADADataset('Train', windowLength, predL, wtNum,dataPath=dataPath)
    scadaTestDataset = SCADADataset('Test', windowLength, predL, wtNum,dataPath=dataPath)



    dataloader = torch.utils.data.DataLoader(scadaTrainDataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(0))

    dataloaderTest = torch.utils.data.DataLoader(scadaTestDataset, batch_size=batch_size,
                                             shuffle=False, num_workers=int(0))
    model = network(WT, P, dropout=dropout,windowLength=windowLength,embedding=False).to(device)

    paras_new = []
    for k, v in dict(model.named_parameters()).items():
        if k == 'embedding_layer1.weight':
            paras_new += [{'params': [v], 'lr': lr * 5}]
        else:
            paras_new += [{'params': [v], 'lr': lr}]


    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=lrG, weight_decay=weight_decayG)
    optimizer = torch.optim.Adam(paras_new, weight_decay=weight_decay)

    scheduler = schedulerFunc(optimizer, 'min', patience=50, verbose=False)

    minW =float('inf')
    minMse=float('inf')
    lossTest=[]
    lossTrain=[]
    for epoch in range(Maxepochs):

        # print(generator.embedding.sftm(generator.embedding.weight))

        model.train()
        yNp = np.zeros([1, 288 * 2])
        yReal = np.zeros([1, 288 * 2])
        for i, (x, y) in enumerate(dataloaderTest):

            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device).view(-1,predL*2)
            y2 = np.copy(y.cpu().detach().numpy())

            ypred = model(x)

            yNp = np.vstack((yNp, ypred.cpu().detach().numpy()))
            yReal = np.vstack((yReal, y2))
            loss = F.mse_loss(y, ypred)
            loss.backward()

            optimizer.step()

            if (i) % (int(len(dataloader)/4)) == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f\t '
                      % (epoch, Maxepochs, i, len(dataloader), np.mean((yNp[1:, :]- yReal[1:, :]) ** 2)))

        scheduler.step(np.mean((yNp[1:, :]- yReal[1:, :]) ** 2))
        lossTrain.append(np.mean((yNp[1:, :]- yReal[1:, :]) ** 2))
        print(np.mean((yNp[1:, :]- yReal[1:, :]) ** 2))
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, '%s/%s.pth' % (outf, name))
        yp, yt =eval(dataloader, model, device)
        lossTest.append(np.mean((yp - yt) ** 2))
        print('testLoss',np.mean((yp - yt) ** 2))

        f, ax = plt.subplots(1, 1)

        ax.plot(lossTrain)
        ax.plot(lossTest)

        f.savefig('%s//loss.png' % (outf), dpi=300, bbox_inches='tight')
        plt.close(f)

if __name__=='__main__':
    outf='E:\YANG Luoxiao\Data\BDWPF'

    train(deviceNum=0, nameInput='LSTM', wtNum=1, outf=outf, schedulerFunc=ReduceLROnPlateau, server=False,
          dataPath='E:\YANG Luoxiao\Data\BDWPF\data')

































































