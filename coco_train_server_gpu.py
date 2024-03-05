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

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
# DEVICE = torch.device("cuda",int(sys.argv[1]))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304]

SEED=SEEDS[int(sys.argv[2])]


np.random.seed(SEED)
torch.manual_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
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
        
        self.fc1=nn.Linear(in_features=4096,out_features=128)
        self.fc2=nn.Linear(in_features=128,out_features=64)
        self.out=nn.Linear(in_features=64,out_features=5)

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
    metric = AUROC(task='multiclass', num_classes=5) 
    # softmax = nn.Softmax(dim=1)
    
    training_accuracy = Accuracy(task='multiclass', num_classes=5).to(DEVICE)
    val_accuracy = Accuracy(task='multiclass', num_classes=5).to(DEVICE)

    torch.manual_seed(SEED)

    for training_ind in [sys.argv[2]]:
        torch.cuda.empty_cache()

        model = Net()
        model = nn.DataParallel(model)
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        
        # Anders et. al uses mainly AdaDelta but also SGD in one section
#         optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)

        # lol
        min_loss = 100000000

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(f' epoch {epoch+1} in {epochs}')
            t0=time.time()
            epoch_loss = 0.0
            epoch_loss_val= 0.0
            batch_acc=[]
            batch_acc_val=[]

            auroc_train=[]
            auroc_val=[]

            for i, data in enumerate(train_loader):
                inputs, labels = data

                inputs = inputs.to(DEVICE,dtype=torch.float)
                labels = labels.type(torch.LongTensor)
                labels=labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs).squeeze()
                _, predicted = torch.max(outputs.data,1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()        
                batch_acc.append(training_accuracy(predicted, labels).item())
                
                auroc_train.append(metric(outputs,labels.to(torch.int32)))

                del inputs, labels, outputs, loss, predicted, data

            auroc_train = sum(auroc_train)/len(auroc_train)

            for i_v, data_val in enumerate(val_loader):
                inputs_val, labels_val = data_val

                inputs_val = inputs_val.to(DEVICE,dtype=torch.float)
                labels_val = labels_val.type(torch.LongTensor)
                labels_val=labels_val.to(DEVICE)

                outputs_val = model(inputs_val).squeeze()
                _, predicted_val = torch.max(outputs_val.data, 1)

                loss_val = criterion(outputs_val, labels_val)

                epoch_loss_val += loss_val.item()        
                batch_acc_val.append(val_accuracy(predicted_val, labels_val).item())

                auroc_val.append(metric(outputs_val,labels_val.to(torch.int32)))

                del inputs_val, labels_val, outputs_val, loss_val, predicted_val, data_val

            auroc_val = sum(auroc_val)/len(auroc_val)

            if epoch_loss_val < min_loss:
                min_loss = epoch_loss_val
                save_model = type(model)() # get a new instance
                # save_model.load_state_dict(model.state_dict()) # copy weights and stuff

                pretrained_dict = {key.replace("module.", "module.module."): value for key, value in model.state_dict().items()}
                save_model.load_state_dict(pretrained_dict) # copy weights and stuff

                torch.save(save_model.state_dict(), f'{str(name_model)}_{training_ind}.pt')
                del save_model

            epoch_acc=sum(batch_acc)/len(batch_acc)
            epoch_acc_val=sum(batch_acc_val)/len(batch_acc_val)

            print(f'epoch train loss: {epoch_loss} | epoch train acc {epoch_acc} | AUROC: {auroc_train}')
            print(f'epoch val loss: {epoch_loss_val} | epoch val acc {epoch_acc_val} | AUROC: {auroc_val}')
            print(f'time elapsed: {round(time.time()-t0,2)} s')

            epoch_loss = 0.0
            epoch_loss_val=0.0
            del epoch_loss_val, auroc_train, auroc_val, epoch_acc, epoch_acc_val
            torch.cuda.empty_cache()

            
        del model, optimizer, criterion
        print('Finished Training')
        print()
        print('\t \t *******************')
        print()
        
        torch.cuda.empty_cache()
    del train_loader, val_loader
def main():
    with open('coco_data.pkl', 'rb') as f:
        [(x_train, y_train, _), (x_val, y_val, _), (_, _, _)] = pickle.load(f)

    train_inds = np.where(y_train==9)[0]
    val_inds = np.where(y_val==9)[0]

    y_train[train_inds] = 4
    y_val[val_inds] = 4
    
    lr = 0.005
    epochs = 50
    momentum = 0.9
    weight_decay = 1e-3
    batch_size = 32

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train.transpose(0,3,1,2)), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val.transpose(0,3,1,2)), torch.tensor(y_val)), batch_size=batch_size, shuffle=True)



    train(lr,epochs,train_loader,val_loader,f'coco_control',momentum=momentum,weight_decay=weight_decay)

if __name__ == '__main__':
    main()
