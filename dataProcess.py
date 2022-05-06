#Author:ike yang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from utlize import  standardizeAndSplitData
wtNumber=134
p=10#ws=0,wp=-1
dataF=pd.read_csv('E:\YANG Luoxiao\Data\BDWPF\clean_data.csv')
data=dataF.values
for i in range(wtNumber):
    if i==0:
        data1 = data[np.where(data[:, 0] == (i+1))[0], 3:].reshape(-1,1,p)
    else:
        data1=np.concatenate((data1,data[np.where(data[:, 0] == (i+1))[0], 3:].reshape(-1,1,p)),axis=1)
print(data1.shape)
dataTrain, dataVal, dataTest, scaler = standardizeAndSplitData(data1,proportion=(153/(15+153),0,15/(15+153)))
with open('E:\YANG Luoxiao\Data\BDWPF\data','wb') as f:
    pickle.dump((dataTrain,dataVal,dataTest,scaler,data1),f)#(26496, 134, 10)