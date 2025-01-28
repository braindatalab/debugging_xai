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
import matplotlib.pyplot as plt
import seaborn as sbs

DEVICE = 'cpu'

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

with open('confounder_test128.pkl', 'rb') as f:
    confounder_test = pickle.load(f)
    confounder_test = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in confounder_test]

with open('suppressor_test128.pkl', 'rb') as f:
    suppressor_test = pickle.load(f)
    suppressor_test = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in suppressor_test]

with open('no_mark_test128.pkl', 'rb') as f:
    no_mark_test = pickle.load(f)
    no_mark_test = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in no_mark_test]

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    return model

def print_AUC(loader, model_ind=0):

    cnn_conf=load_trained(f'./models/cnn_conf_{model_ind}.pt').to(DEVICE).eval()
    cnn_sup=load_trained(f'./models/cnn_sup_{model_ind}.pt').to(DEVICE).eval()
    cnn_no=load_trained(f'./models/cnn_no_{model_ind}.pt').to(DEVICE).eval()
    
    #models -> cnn_no; cnn_sup; cnn_conf

    softmax = nn.Softmax(dim=1)
    # metric = AUROC(num_classes=2, task='binary')

    metric_conf = AUROC(num_classes=2, task='binary')
    metric_sup = AUROC(num_classes=2, task='binary')
    metric_no = AUROC(num_classes=2, task='binary')

    model_conf_test_conf_pred=np.array([])
    model_sup_test_conf_pred=np.array([])
    model_no_test_conf_pred=np.array([])
    true_labels=np.array([])
    with torch.no_grad():
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

            acc_conf = metric_conf(out_pred_conf[:,1], labels_test)
            acc_sup = metric_sup(out_pred_sup[:,1], labels_test)
            acc_no = metric_no(out_pred_no[:,1], labels_test)
        
            model_conf_test_conf_pred=np.concatenate((model_conf_test_conf_pred, out_pred_conf[:,1].cpu().detach().numpy()))
            model_sup_test_conf_pred=np.concatenate((model_sup_test_conf_pred, out_pred_sup[:,1].cpu().detach().numpy()))
            model_no_test_conf_pred=np.concatenate((model_no_test_conf_pred, out_pred_no[:,1].cpu().detach().numpy()))
            true_labels=np.concatenate((true_labels, labels_test.cpu().numpy()))

        acc_conf = metric_conf.compute()
        acc_sup = metric_sup.compute()
        acc_no = metric_no.compute()

        print('model: confounder',acc_conf)
        print('model: suppressor',acc_sup)
        print('model: no watermark',acc_no)
        print()

        l=[acc_conf,
        acc_sup,
        acc_no]
        return l, [model_conf_test_conf_pred, model_sup_test_conf_pred, model_no_test_conf_pred, true_labels]


batch_size=64
confounder_test_loader = DataLoader(confounder_test,batch_size=batch_size, shuffle=False)
suppressor_test_loader = DataLoader(suppressor_test,batch_size=batch_size, shuffle=False)
no_mark_test_loader = DataLoader(no_mark_test,batch_size=batch_size, shuffle=False)

confounder_data_results = []
suppressor_data_results = []
no_mark_data_results = []

conf_conf = np.array([])
conf_sup = np.array([])
conf_no = np.array([])

sup_conf = np.array([])
sup_sup = np.array([])
sup_no = np.array([])

no_conf = np.array([])
no_sup = np.array([])
no_no = np.array([])

conf_labels = np.array([])
sup_labels = np.array([])
no_labels = np.array([])

for i in range(10):
    SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

    SEED=SEEDS[i] 


    np.random.seed(SEED)
    torch.manual_seed(SEED)

    import os
    os.environ['PYTHONHASHSEED']=str(SEED)

    import random
    random.seed(SEED)

    print(f'MODEL {i}')
    print('confounder data:')

    conf_scores, [conf_model_conf_test, conf_model_sup_test, conf_model_no_test, conf_true_labels] = print_AUC(confounder_test_loader, i)

    print()
    print('suppressor data:')
    sup_scores, [sup_model_conf_test, sup_model_sup_test, sup_model_no_test, sup_true_labels] = print_AUC(suppressor_test_loader, i)

    print()
    print('no watermark data:')
    no_scores, [no_model_conf_test, no_model_sup_test, no_model_no_test, no_true_labels] = print_AUC(no_mark_test_loader, i)

    confounder_data_results.append(conf_scores)

    suppressor_data_results.append(sup_scores)

    no_mark_data_results.append(no_scores)
    print()


    conf_conf = np.concatenate((conf_conf, conf_model_conf_test))
    conf_sup = np.concatenate((conf_sup, conf_model_sup_test))
    conf_no = np.concatenate((conf_no, conf_model_no_test))
    
    sup_conf = np.concatenate((sup_conf, sup_model_conf_test))
    sup_sup = np.concatenate((sup_sup, sup_model_sup_test))
    sup_no = np.concatenate((sup_no, sup_model_no_test))
    
    no_conf = np.concatenate((no_conf, no_model_conf_test))
    no_sup = np.concatenate((no_sup, no_model_sup_test))
    no_no = np.concatenate((no_no, no_model_no_test))

    conf_labels = np.concatenate((conf_labels, conf_true_labels))
    sup_labels = np.concatenate((sup_labels, sup_true_labels))
    no_labels = np.concatenate((no_labels, no_true_labels))
    
print('MEAN AUROC')
print(torch.mean(torch.tensor(confounder_data_results), axis=0))
print(torch.mean(torch.tensor(suppressor_data_results), axis=0))
print(torch.mean(torch.tensor(no_mark_data_results), axis=0))

print('STD AUROC')
print(torch.std(torch.tensor(confounder_data_results), axis=0))
print(torch.std(torch.tensor(suppressor_data_results), axis=0))
print(torch.std(torch.tensor(no_mark_data_results), axis=0))


import pickle as pkl

with open('auroc_conf.pickle', 'wb') as f:
    pickle.dump([conf_conf, conf_sup, conf_no], f)    

with open('auroc_sup.pickle', 'wb') as f:
    pickle.dump([sup_conf, sup_sup, sup_no], f)

with open('auroc_no.pickle', 'wb') as f:
    pickle.dump([no_conf, no_sup, no_no], f)

with open('conf_labels.pickle', 'wb') as f:
    pickle.dump(conf_labels, f)

with open('sup_labels.pickle', 'wb') as f:
    pickle.dump(sup_labels, f)

with open('no_labels.pickle', 'wb') as f:
    pickle.dump(no_labels, f)

from sklearn.metrics import roc_curve

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

models = ['Confounder', 'Suppressor', 'No Watermark']
colours = ['r', 'g', 'b']

for i, (test_labels, results) in enumerate([[conf_labels, [conf_conf, sup_conf, no_conf]], [sup_labels, [conf_sup, sup_sup, no_sup]], [no_labels, [conf_no, sup_no, no_no]]]):
    for j, result in enumerate(results):
        fpr, tpr, _ = roc_curve(test_labels, result)
        axs[i].plot(fpr, tpr, color=colours[j], label=f'{models[j]} Model')
        
        # axs[i,j].axis('off')
    axs[i].set_title(f'{models[i]} Dataset')
    axs[i].set_xlabel('False Positive Rate')
    axs[i].plot([0, 1], [0, 1], 'k--') 


axs[0].set_ylabel('True Positive Rate')

# Add legend
axs[2].legend(loc='lower right')

plt.tight_layout()
plt.savefig('roc_curves.png', bbox_inches='tight')