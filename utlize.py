#Author:ike yang
from sklearn import preprocessing
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch
def standardizeAndSplitData(data,splitType='minmax',proportion=(0.6,0.2,0.2),minmaxOnly=False ):
    if splitType=='minmax':
        scaler = preprocessing.MinMaxScaler()
    elif splitType=='zeroscore':
        scaler = preprocessing.StandardScaler()
    t,wt,p=data.shape
    if minmaxOnly:
        data=scaler.fit_transform(data).reshape(-1, wt, p)
        return data,scaler
    daraTrain=data[:int(t*proportion[0]),:,:].reshape(-1,p)
    daraVal=data[int(t*proportion[0]):int(t*(proportion[0]+proportion[1])),:,:].reshape(-1,p)
    daraTest=data[int(t*(proportion[0]+proportion[1])):,:,:].reshape(-1,p)

    daraTrain=scaler.fit_transform(daraTrain).reshape(-1,wt,p)
    daraTest=scaler.transform(daraTest).reshape(-1,wt,p)
    if proportion[1]==0:
        return daraTrain, None, daraTest, scaler
    daraVal=scaler.transform(daraVal).reshape(-1,wt,p)
    return daraTrain,daraVal,daraTest,scaler


class SCADADataset(Dataset):
    def __init__(self, typeData, windowLength, predL, wtnum,dataPath='E:\YANG Luoxiao\Data\BDWPF\data'):
        with open(dataPath,'rb') as f:
            dataTrain, dataVal, dataTest, scaler, data1 = pickle.load(f)
        self.predL = predL
        self.windL = windowLength
        self.wtnum = wtnum
        if typeData=='Train':
            self.data=dataTrain
        if typeData=='Eval':
            self.data=dataVal
        if typeData=='Test':
            self.data=dataTest

        self.datashape = list(self.data.shape)



    def __len__(self):
        return self.data.shape[0] - self.windL - self.predL

    def __getitem__(self, idx):

        x = np.copy(self.data[idx:idx + self.windL, :, :])
        x = torch.from_numpy(x).float()
        y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, [0,-1]])
        y = torch.from_numpy(y).float()
        return x, y






















