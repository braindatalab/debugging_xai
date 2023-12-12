import torch
from torch import manual_seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy,AUROC


import pickle
import time
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sbs

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
DEVICE = torch.device("cuda",int(sys.argv[2]))



def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

SEED=1234
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    metric = AUROC(task='binary', num_classes=2) 
    softmax = nn.Softmax(dim=1)
    
    training_accuracy = Accuracy(task='binary').to(DEVICE)
    val_accuracy = Accuracy(task='binary').to(DEVICE)

    manual_seed(SEED)

    trained_models = []
    training_logs = []

    for training_ind in range(10):
        cnn_conf=Net()
        cnn_conf.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn_conf.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


        loss_list_train=[]
        loss_list_val=[]
        acc_list_train=[]
        acc_list_val=[]
        

        min_loss = 1000000
    #     last_loss = 1000000000

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(f' epoch {epoch+1} in {epochs}')
            t0=time.time()
            epoch_loss = 0.0
            epoch_loss_val= 0.0
            batch_acc=[]
            batch_acc_val=[]

            auroc_train=[]
            auroc_val=[]

            pred_train=[]
            pred_val=[]
            true_labels_train=[]
            true_labels_val=[]

            for i, data in enumerate(train_loader):
                inputs, labels = data

                inputs = inputs.to(DEVICE,dtype=torch.float)
                labels = labels.type(torch.LongTensor)
                labels=labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = cnn_conf(inputs).squeeze()
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()        
                batch_acc.append(training_accuracy(predicted, labels).item())

                pred_train=np.concatenate((pred_train, softmax(outputs)[:,1].cpu().detach().numpy()))
                true_labels_train=np.concatenate((true_labels_train, labels.cpu().numpy()))

            auroc_train=metric(torch.tensor(pred_train),torch.tensor(true_labels_train).to(torch.int32))

            for i_v, data_val in enumerate(val_loader):
                inputs_val, labels_val = data_val

                inputs_val = inputs_val.to(DEVICE,dtype=torch.float)
                labels_val = labels_val.type(torch.LongTensor)
                labels_val=labels_val.to(DEVICE)

                outputs_val = cnn_conf(inputs_val).squeeze()
                _, predicted_val = torch.max(outputs_val.data, 1)

                loss_val = criterion(outputs_val, labels_val)

                epoch_loss_val += loss_val.item()        
                batch_acc_val.append(val_accuracy(predicted_val, labels_val).item())


                pred_val=np.concatenate((pred_val, softmax(outputs_val)[:,1].cpu().detach().numpy()))
                true_labels_val=np.concatenate((true_labels_val, labels_val.cpu().detach().numpy()))

            if epoch_loss_val < min_loss:
                min_loss = epoch_loss_val
                save_model = type(cnn_conf)() # get a new instance
                save_model.load_state_dict(cnn_conf.state_dict()) # copy weights and stuff

                torch.save(save_model.state_dict(), f'{str(name_model)}_{training_ind}.pt')

            auroc_val=metric(torch.tensor(pred_val),torch.tensor(true_labels_val).to(torch.int32))

            epoch_acc=sum(batch_acc)/len(batch_acc)
            epoch_acc_val=sum(batch_acc_val)/len(batch_acc_val)

            print(f'epoch train loss: {epoch_loss} | epoch train acc {epoch_acc} | AUROC: {auroc_train}')
            print(f'epoch val loss: {epoch_loss_val} | epoch val acc {epoch_acc_val} | AUROC: {auroc_val}')
            print(f'time elapsed: {round(time.time()-t0,2)} s')

            loss_list_train.append(epoch_loss)
            loss_list_val.append(epoch_loss_val)
            acc_list_train.append(epoch_acc)
            acc_list_val.append(epoch_acc_val)

            epoch_loss = 0.0
            epoch_loss_val=0.0

        print('Finished Training')
        print()
        print('\t \t *******************')
        print()
        
        torch.save(save_model.state_dict(), f'{str(name_model)}_{training_ind}.pt')

        trained_models.append(save_model)
        training_logs.append([acc_list_train, loss_list_train, acc_list_val, loss_list_val])
#     plt.plot(loss_list_train)
#     plt.plot(loss_list_val)
#     plt.plot([i*100 for i in acc_list_train])
#     plt.plot([i*100 for i in acc_list_va])
    return trained_models, [acc_list_train, loss_list_train, acc_list_val, loss_list_val]

# Training vars
batch_size=64
manual_seed(SEED)
lr=0.005
epochs=100
weight_decay=0.001


with open('confounder_train128.pkl', 'rb') as f:
    confounder_train = pickle.load(f)
    confounder_train = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in confounder_train]
print('train')

with open('confounder_val128.pkl', 'rb') as f:
    confounder_val = pickle.load(f)
    confounder_val = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in confounder_val]
print('val')

with open('confounder_test128.pkl', 'rb') as f:
    confounder_test = pickle.load(f)
    confounder_test = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in confounder_test]
print('test')


confounder_train_loader = DataLoader(confounder_train,batch_size=batch_size, shuffle=True)
confounder_val_loader = DataLoader(confounder_val,batch_size=batch_size, shuffle=True)
# confounder_hold_loader = DataLoader(confounder_test,batch_size=batch_size, shuffle=False)

conf_models, conf_logs = train(lr,epochs,confounder_train_loader,confounder_val_loader,'cnn_conf',weight_decay=weight_decay)


with open('suppressor_train128.pkl', 'rb') as f:
    suppressor_train = pickle.load(f)
    suppressor_train = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in suppressor_train]
print('train')

with open('suppressor_validation128.pkl', 'rb') as f:
    suppressor_val = pickle.load(f)
    suppressor_val = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in suppressor_val]
print('val')

with open('suppressor_test128.pkl', 'rb') as f:
    suppressor_test = pickle.load(f)
    suppressor_test = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in suppressor_test]
print('test')


suppressor_train_loader = DataLoader(suppressor_train,batch_size=batch_size, shuffle=True)
suppressor_val_loader = DataLoader(suppressor_val,batch_size=batch_size, shuffle=True)
# suppressor_hold_loader = DataLoader(suppressor_test,batch_size=batch_size, shuffle=False)

cnn_sup=train(lr,epochs,suppressor_train_loader,suppressor_val_loader,'cnn_sup',weight_decay=weight_decay)

with open('no_mark_train128.pkl', 'rb') as f:
    no_mark_train = pickle.load(f)
    no_mark_train = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in no_mark_train]
print('train')

with open('no_mark_validation128.pkl', 'rb') as f:
    no_mark_val = pickle.load(f)
    no_mark_val = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in no_mark_val]
print('val')

with open('no_mark_test128.pkl', 'rb') as f:
    no_mark_test = pickle.load(f)
    no_mark_test = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in no_mark_test]
print('test')


no_mark_train_loader = DataLoader(no_mark_train,batch_size=batch_size, shuffle=True)
no_mark_val_loader = DataLoader(no_mark_val,batch_size=batch_size, shuffle=True)
# no_mark_test_loader = DataLoader(no_mark_test,batch_size=batch_size, shuffle=False)

cnn_no=train(lr,epochs,no_mark_train_loader,no_mark_val_loader,'cnn_no',weight_decay=weight_decay)
