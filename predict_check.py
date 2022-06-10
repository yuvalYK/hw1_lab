import torch
import torch.nn as nn

#from torch.optim import Adam

import os
import numpy as np
import pandas as pd
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
#from sklearn.linear_model import LinearRegression
import csv
import sys

class net(nn.Module):
    def __init__(self, size, n_1, n_2,n_3):
        super(net, self).__init__()
        
        self.l1 = nn.Linear(size, n_1)
        self.l2 = nn.Linear(n_1, n_2)
        self.l3 = nn.Linear(n_2, n_3)
        self.l4 = nn.Linear(n_3,1)
        
        #self.drop = nn.Dropout(p=d)
        self.act = nn.ReLU()
        self.soft = nn.Sigmoid()
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(self.act(x))
        x = self.l3(self.act(x))
        x = self.l4(self.act(x))
        
        return self.soft(x)
    
#print(os.getcwd())
#print(sys.path[0])
#print(sys.path[0]+'model_relu2.pth')
    
def clip_df(df, label='SepsisLabel'):
    df = df.drop(labels=['Unit1', 'Unit2','EtCO2','Bilirubin_direct','TroponinI','Fibrinogen'], axis=1)
    index = df.loc[df[label] == 1].index
    if len(index) == 0:
        return df
    else:
        return df.loc[: index[0]]

def main():
    if len(sys.argv)==1:
        print('no file path given')
        return 0
    
    #try:
    #    dicti = torch.load(sys.path[0]+'/model_relu2.pth')
    #except:
    #    print('cannot load model. please make sure model.pth is in directory')
    #    return 0
    
    data_path = sys.argv[1]
    
    if data_path[-1] != '/':
        data_path = data_path+'/'
    
    #print(data_path)
    
    data_list = os.listdir(data_path)
    #print(data_list)
    
    if data_list[0].split('_')[0] != 'patient':
        data_path = data_path+'test/'
        data_list = os.listdir(data_path)
    
    #print(data_list)
    
    df_list = []
    pat_list = []
    for i in data_list:
        df_temp = pd.read_csv(data_path+i,sep='|')
        df_list.append(df_temp)
        pat_list.append(i.split('_')[1].split('.')[0])
    
    new_df_list = []
    for df in df_list:
        df_new = clip_df(df)
        #print(df_new)
        new_df_list.append(df_new)
        #break
    
    #coef = dicti['coef']
    
    y = []
    X = []
    for df in new_df_list:
        label = df.iloc[-1]['SepsisLabel']
        #means = df.mean().to_dict()

        #df_nu = (1-df.isna().mean())
        #df_nu = df_nu.drop(labels=['Age', 'Gender','HospAdmTime','ICULOS','SepsisLabel'])

        #for key in means.keys():
        #    if np.isnan(means[key]):
        #        means[key] = coef[means['Gender']][key][0]*means['Age'] + coef[means['Gender']][key][1]

        y.append(int(label))
        #means.pop('SepsisLabel',None)
        #means_list = list(means.values())
        #means_list.extend(list(df_nu))
        #X.append(means_list)
        
    #x = torch.Tensor(X)
    #y = torch.Tensor(y)
    y = np.array(y)
    y_test = []
    with open('prediction.csv') as f:
        reader = csv.reader(f)
        
        for line in reader:
            y_test.append(int(line[1]))
    
    #print(type(y[0]), type(y_test[0]))
    
    print(f1_score(y,y_test))
    
    #n_log = net(dicti['hyper']['size'], dicti['hyper']['layer 1'], dicti['hyper']['layer 2'], dicti['hyper']['layer 3'])
    
    #n_log.load_state_dict(dicti['params'])
    
    #n_log.eval()
    
    #y_pred = n_log(x)
    
    #lis_csv_fin = []
    #for i in range(len(y_pred)):
    #    lis_csv_fin.append([int(pat_list[i]), 0 if y_pred[i].item() < 0.5 else 1])
    #lis_csv_fin = sorted(lis_csv_fin, key=lambda x: x[0])
    
    #with open('prediction.csv', 'w', newline='') as f:
    #    writer = csv.writer(f)
        #writer.writerow(['id', 'SepsisLabel'])
        
    #    for x in lis_csv_fin:
    #        writer.writerow([x[0], x[1]])
if __name__ == "__main__":
    main()