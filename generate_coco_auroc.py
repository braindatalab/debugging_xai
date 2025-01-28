import torch
from torch import manual_seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy,AUROC

import pickle
import time
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sbs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
# DEVICE = torch.device("cuda",int(sys.argv[1]))

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5, padding='same')
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding='same')
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5, padding='same')
        
        self.maxPooling2=nn.MaxPool2d(kernel_size=2)
        self.maxPooling4_0=nn.MaxPool2d(kernel_size=4)
        self.maxPooling4_1=nn.MaxPool2d(kernel_size=4)
#         self.adPooling=nn.AdaptiveAvgPool1d(256)
        
        self.fc1=nn.Linear(in_features=12544,out_features=128)
        self.fc2=nn.Linear(in_features=128,out_features=64)
        self.out=nn.Linear(in_features=64,out_features=2)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxPooling4_0(x)
        x=F.relu(x)
        
        x=self.conv2(x)
        x=self.maxPooling4_1(x)
        x=F.relu(x)
        
        x=self.conv3(x)
        x=self.maxPooling2(x)
        x=F.relu(x)
        
        x=F.dropout(x)
        x=x.view(1,x.size()[0],-1) #stretch to 1d data
        #x=self.adPooling(x).squeeze()
        
        x=self.fc1(x)
        x=F.relu(x)
        
        x=self.fc2(x)
        x=F.relu(x)
        
        x=self.out(x)
        
        return x[0]

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    return model

def print_AUC(loader, model_ind=0):

    cnn_conf=load_trained(f'./coco_conf_{model_ind}.pt').to(DEVICE)
    cnn_sup=load_trained(f'./coco_sup_{model_ind}.pt').to(DEVICE)
    cnn_no=load_trained(f'./coco_norm_{model_ind}.pt').to(DEVICE)
    
    #models -> cnn_no; cnn_sup; cnn_conf

    softmax = nn.Softmax(dim=1)
    metric_conf = AUROC(num_classes=2, task='binary')
    metric_sup = AUROC(num_classes=2, task='binary')
    metric_no = AUROC(num_classes=2, task='binary')

    for i_v, data_test in enumerate(loader):
        inputs_test, labels_test = data_test
        inputs_test = inputs_test.to(DEVICE,dtype=torch.float)
        labels_test = labels_test.type(torch.LongTensor)
        labels_test=labels_test.to(DEVICE)

        outputs_conf = cnn_conf(inputs_test).squeeze()
        outputs_sup = cnn_sup(inputs_test).squeeze()
        outputs_no = cnn_no(inputs_test).squeeze()

        out_pred_conf=softmax(outputs_conf)
        out_pred_sup=softmax(outputs_sup)
        out_pred_no=softmax(outputs_no)

        metric_conf.update(torch.tensor(out_pred_conf[:,1].cpu().detach().numpy()),torch.tensor(labels_test.cpu().numpy()))
        metric_sup.update(torch.tensor(out_pred_sup[:,1].cpu().detach().numpy()),torch.tensor(labels_test.cpu().numpy()))
        metric_no.update(torch.tensor(out_pred_no[:,1].cpu().detach().numpy()),torch.tensor(labels_test.cpu().numpy()))

    auroc_conf = metric_conf.compute()
    auroc_sup = metric_sup.compute()
    auroc_no = metric_no.compute()

    print('model: confounder',auroc_conf)
    print('model: suppressor',auroc_sup)
    print('model: norm',auroc_no)
    print()
    
    return [auroc_conf, auroc_sup, auroc_no]



batch_size=32

with open(f'coco_data_norm.pkl', 'rb') as f:
    [(_, _, _), (_, _, _), (x_test, y_test, _)] = pickle.load(f)

norm_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=False, num_workers=4)

with open(f'coco_data_conf.pkl', 'rb') as f:
    [(_, _, _), (_, _, _), (x_test, y_test, _)] = pickle.load(f)

conf_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=False, num_workers=4)

with open(f'coco_data_sup.pkl', 'rb') as f:
    [(_, _, _), (_, _, _), (x_test, y_test, _)] = pickle.load(f)

sup_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=False, num_workers=4)

confounder_data_results = []
suppressor_data_results = []
no_mark_data_results = []

for i in range(1):
    SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304]

    SEED=SEEDS[0] 


    np.random.seed(SEED)
    torch.manual_seed(SEED)

    import os
    os.environ['PYTHONHASHSEED']=str(SEED)

    import random
    random.seed(SEED)

    print(f'MODEL {i}')
    print('confounder data:')
    confounder_data_results.append(print_AUC(conf_loader, i))
    print()
    print('suppressor data:')
    suppressor_data_results.append(print_AUC(sup_loader, i))
    print()
    print('norm data:')
    no_mark_data_results.append(print_AUC(norm_loader, i))
    print()
