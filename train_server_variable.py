import torch
from torch import manual_seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy,AUROC

import os
import random

import pickle
import time
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sbs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
# DEVICE = torch.device("cuda",int(sys.argv[1]))
# DEVICE = 'cpu'


def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

SEED=SEEDS[1] 

manual_seed(SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5)
        
        self.maxPooling2=nn.MaxPool2d(kernel_size=2)
        self.maxPooling4_0=nn.MaxPool2d(kernel_size=4)
        self.maxPooling4_1=nn.MaxPool2d(kernel_size=4)
#         self.adPooling=nn.AdaptiveAvgPool1d(256)
        
        self.fc1=nn.Linear(in_features=256,out_features=128)
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


def train(lr,epochs,train_loader,val_loader,name_model,momentum=0.9,weight_decay=1e-3):
    for training_ind in range(10):
        SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

        SEED=SEEDS[training_ind] 

        manual_seed(SEED)

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        os.environ['PYTHONHASHSEED']=str(SEED)

        random.seed(SEED)

        metric = AUROC(task='multiclass', num_classes=2) 
        metric_val = AUROC(task='multiclass', num_classes=2) 
        # softmax = nn.Softmax(dim=1)
        
        training_accuracy = Accuracy(task='multiclass', num_classes=2).to(DEVICE)
        val_accuracy = Accuracy(task='multiclass', num_classes=2).to(DEVICE)

        model=Net()
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)        

        min_loss = 1000000

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(f' epoch {epoch+1} in {epochs}')
            t0=time.time()
            epoch_loss = 0.0
            epoch_loss_val= 0.0

            for i, data in enumerate(train_loader):
                inputs, labels = data

                inputs = inputs.to(DEVICE,dtype=torch.float)
                labels = labels.type(torch.LongTensor)
                labels=labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs).squeeze()
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()        

                training_accuracy.update(predicted, labels)
                metric.update(outputs, labels.to(torch.int32))

            model.eval()
            with torch.no_grad():
                for i_v, data_val in enumerate(val_loader):
                    inputs_val, y_val = data_val

                    inputs_val = inputs_val.to(DEVICE,dtype=torch.float)
                    y_val = y_val.type(torch.LongTensor)
                    y_val=y_val.to(DEVICE)

                    outputs_val = model(inputs_val).squeeze()
                    _, predicted_val = torch.max(outputs_val.data, 1)

                    loss_val = criterion(outputs_val, y_val)

                    epoch_loss_val += loss_val.item() 

                    val_accuracy.update(predicted_val, y_val)
                    metric_val.update(outputs_val, y_val.to(torch.int32))
            model.train()

            auroc_train = metric.compute()
            auroc_val = metric_val.compute()

            epoch_acc = training_accuracy.compute()
            epoch_acc_val = val_accuracy.compute()

            if epoch_loss_val < min_loss:
                min_loss = epoch_loss_val
                save_model = type(model)() # get a new instance
                save_model.load_state_dict(model.state_dict()) # copy weights and stuff

                torch.save(save_model.state_dict(), f'{str(name_model)}_{training_ind}.pt')



            print(f'epoch train loss: {epoch_loss} | epoch train acc {epoch_acc} | AUROC: {auroc_train}')
            print(f'epoch val loss: {epoch_loss_val} | epoch val acc {epoch_acc_val} | AUROC: {auroc_val}')
            print(f'time elapsed: {round(time.time()-t0,2)} s')

            training_accuracy.reset()
            val_accuracy.reset()

            metric.reset()
            metric_val.reset()

            epoch_loss = 0.0
            epoch_loss_val=0.0

        print('Finished Training')
        print()
        print('\t \t *******************')
        print()



# Training vars
batch_size=64
lr=0.005
epochs=100
weight_decay=0.001

if len(sys.argv) < 3 :
    print('Usage: python train_server_gpu.py <gpu_id> <conf/sup/no> <split>')
    sys.exit(1)


for dataset in ['confounder','suppressor','no_watermark']:
    if sys.argv[1] in dataset or sys.argv[1] == 'all':
        with open(f'./artifacts/split_{sys.argv[2]}_{dataset}_variable_train.pkl', 'rb') as f:
            x_train, y_train, _, _ = pickle.load(f)
            x_train = [[rescale_values(x,1,0).transpose(2,0,1),y_train.flatten()[i]] for i,x in enumerate(x_train)]
        print('train')

        with open(f'./artifacts/split_{sys.argv[2]}_{dataset}_variable_val.pkl', 'rb') as f:
            x_val, y_val, _, _ = pickle.load(f)
            x_val = [[rescale_values(x,1,0).transpose(2,0,1),y_val.flatten()[i]] for i,x in enumerate(x_val)]
        print('val')


        x_train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
        x_val_loader = DataLoader(x_val, batch_size=batch_size, shuffle=True)

        train(lr,epochs,x_train_loader,x_val_loader,f'./models/cnn_{dataset}_variable_{sys.argv[2]}',weight_decay=weight_decay)
        print(f'training done for {dataset} split number {sys.argv[2]}')
