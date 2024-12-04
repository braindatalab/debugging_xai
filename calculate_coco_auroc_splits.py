import torch
from torch import manual_seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy,AUROC

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

# def print_AUC(loader, model_ind=0):

#     cnn_conf=load_trained(f'./models/coco_conf_{model_ind}.pt').eval().to(DEVICE)
#     cnn_sup=load_trained(f'./models/coco_sup_{model_ind}.pt').eval().to(DEVICE)
#     cnn_no=load_trained(f'./models/coco_norm_{model_ind}.pt').eval().to(DEVICE)
    
#     #models -> cnn_no; cnn_sup; cnn_conf

#     softmax = nn.Softmax(dim=1)
#     metric_conf = AUROC(num_classes=2, task='binary')
#     metric_sup = AUROC(num_classes=2, task='binary')
#     metric_no = AUROC(num_classes=2, task='binary')

#     for i_v, data_test in enumerate(loader):
#         inputs_test, labels_test = data_test
#         inputs_test = inputs_test.to(DEVICE,dtype=torch.float)
#         labels_test = labels_test.type(torch.LongTensor)
#         labels_test=labels_test.to(DEVICE)

#         outputs_conf = cnn_conf(inputs_test).squeeze()
#         outputs_sup = cnn_sup(inputs_test).squeeze()
#         outputs_no = cnn_no(inputs_test).squeeze()

#         out_pred_conf=softmax(outputs_conf)
#         out_pred_sup=softmax(outputs_sup)
#         out_pred_no=softmax(outputs_no)

#         metric_conf.update(torch.tensor(out_pred_conf[:,1].cpu().detach().numpy()),torch.tensor(labels_test.cpu().numpy()))
#         metric_sup.update(torch.tensor(out_pred_sup[:,1].cpu().detach().numpy()),torch.tensor(labels_test.cpu().numpy()))
#         metric_no.update(torch.tensor(out_pred_no[:,1].cpu().detach().numpy()),torch.tensor(labels_test.cpu().numpy()))

#     auroc_conf = metric_conf.compute()
#     auroc_sup = metric_sup.compute()
#     auroc_no = metric_no.compute()

#     print('model: confounder',auroc_conf)
#     print('model: suppressor',auroc_sup)
#     print('model: norm',auroc_no)
#     print()
    
#     return [auroc_conf, auroc_sup, auroc_no]

def print_AUC(loader, split=0, model_ind=0):

    cnn_conf=load_trained(f'./models/coco_conf_{split}_{model_ind}.pt').eval().to(DEVICE)
    cnn_sup=load_trained(f'./models/coco_sup_{split}_{model_ind}.pt').eval().to(DEVICE)
    cnn_no=load_trained(f'./models/coco_norm_{split}_{model_ind}.pt').eval().to(DEVICE)
    
    #models -> cnn_no; cnn_sup; cnn_conf

    softmax = nn.Softmax(dim=1)
    # metric = AUROC(num_classes=2, task='binary')

    metric_conf = AUROC(num_classes=2, task='binary')
    metric_sup = AUROC(num_classes=2, task='binary')
    metric_no = AUROC(num_classes=2, task='binary')

    model_conf_test_conf_pred=np.array([])
    model_sup_test_conf_pred=np.array([])
    model_norm_test_conf_pred=np.array([])
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
            model_norm_test_conf_pred=np.concatenate((model_norm_test_conf_pred, out_pred_no[:,1].cpu().detach().numpy()))
            true_labels=np.concatenate((true_labels, labels_test.cpu().numpy()))

        acc_conf = metric_conf.compute()
        acc_sup = metric_sup.compute()
        acc_no = metric_no.compute()

        print('model: confounder',acc_conf)
        print('model: suppressor',acc_sup)
        print('model: norm',acc_no)
        print()

        l=[acc_conf,
        acc_sup,
        acc_no]
        return l, [model_conf_test_conf_pred, model_sup_test_conf_pred, model_norm_test_conf_pred, true_labels]


batch_size=32

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

SEED=SEEDS[1] 

np.random.seed(SEED)
torch.manual_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

split = int(sys.argv[1])

with open(f'./artifacts/split_{split}_coco_data_norm_test.pkl', 'rb') as f:
    [x_test, y_test, _, _] = pickle.load(f)

norm_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=True, num_workers=4)

with open(f'./artifacts/split_{split}_coco_data_conf_test.pkl', 'rb') as f:
    [x_test, y_test, _, _] = pickle.load(f)

confounder_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=True, num_workers=4)

with open(f'./artifacts/split_{split}_coco_data_sup_test.pkl', 'rb') as f:
    [x_test, y_test, _, _] = pickle.load(f)
    
suppressor_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=True, num_workers=4)


confounder_data_results = []
suppressor_data_results = []
norm_data_results = []

conf_conf = np.array([])
conf_sup = np.array([])
conf_no = np.array([])

sup_conf = np.array([])
sup_sup = np.array([])
sup_no = np.array([])

norm_conf = np.array([])
norm_sup = np.array([])
norm_no = np.array([])

conf_labels = np.array([])
sup_labels = np.array([])
norm_labels = np.array([])

# for split in [int(sys.argv[1])]:



for i in range(5):
    SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

    SEED=SEEDS[i] 

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    import os
    os.environ['PYTHONHASHSEED']=str(SEED)

    import random
    random.seed(SEED)

    print(f'SPLIT {split} MODEL {i}')
    conf_scores, [conf_model_conf_test, sup_model_conf_test, norm_model_conf_test, conf_true_labels] = print_AUC(confounder_loader, split, i)
    print()
    print('suppressor data:')
    sup_scores, [conf_model_sup_test, sup_model_sup_test, norm_model_sup_test, sup_true_labels] = print_AUC(suppressor_loader, split, i)

    print()
    print('norm data:')
    norm_scores, [conf_model_norm_test, sup_model_norm_test, norm_model_norm_test, norm_true_labels] = print_AUC(norm_loader, split, i)

    confounder_data_results.append(conf_scores)

    suppressor_data_results.append(sup_scores)

    norm_data_results.append(norm_scores)
    print()

    conf_conf = np.concatenate((conf_conf, conf_model_conf_test))
    conf_sup = np.concatenate((conf_sup, conf_model_sup_test))
    conf_no = np.concatenate((conf_no, conf_model_norm_test))
    
    sup_conf = np.concatenate((sup_conf, sup_model_conf_test))
    sup_sup = np.concatenate((sup_sup, sup_model_sup_test))
    sup_no = np.concatenate((sup_no, sup_model_norm_test))
    
    norm_conf = np.concatenate((norm_conf, norm_model_conf_test))
    norm_sup = np.concatenate((norm_sup, norm_model_sup_test))
    norm_no = np.concatenate((norm_no, norm_model_norm_test))

    conf_labels = np.concatenate((conf_labels, conf_true_labels))
    sup_labels = np.concatenate((sup_labels, sup_true_labels))
    norm_labels = np.concatenate((norm_labels, norm_true_labels))

print('                        MODEL   ')
print('MEAN AUROC            CONF    SUP    NO')
print('conf data: ', torch.mean(torch.tensor(confounder_data_results), axis=0))
print('sup data: ', torch.mean(torch.tensor(suppressor_data_results), axis=0))
print('norm data: ', torch.mean(torch.tensor(norm_data_results), axis=0))

print('STD AUROC')
print('conf data: ', torch.std(torch.tensor(confounder_data_results), axis=0))
print('sup data: ', torch.std(torch.tensor(suppressor_data_results), axis=0))
print('norm data: ', torch.std(torch.tensor(norm_data_results), axis=0))

import pickle as pkl
split = sys.argv[1]

with open(f'./aurocs/auroc_results_coco_{split}.pickle', 'wb') as f:
    pickle.dump([confounder_data_results, suppressor_data_results, norm_data_results], f)

with open(f'./aurocs/auroc_conf_coco_{split}.pickle', 'wb') as f:
    pickle.dump([conf_conf, conf_sup, conf_no], f)    

with open(f'./aurocs/auroc_sup_coco_{split}.pickle', 'wb') as f:
    pickle.dump([sup_conf, sup_sup, sup_no], f)

with open(f'./aurocs/auroc_norm_coco_{split}.pickle', 'wb') as f:
    pickle.dump([norm_conf, norm_sup, norm_no], f)

with open(f'./aurocs/conf_labels_coco_{split}.pickle', 'wb') as f:
    pickle.dump(conf_labels, f)

with open(f'./aurocs/sup_labels_coco_{split}.pickle', 'wb') as f:
    pickle.dump(sup_labels, f)

with open(f'./aurocs/norm_labels_coco_{split}.pickle', 'wb') as f:
    pickle.dump(norm_labels, f)


fig, axs = plt.subplots(1, 3, figsize=(18, 6))

models = ['Confounder', 'Suppressor', 'No Watermark']
colours = ['r', 'g', 'b']

for i, (test_labels, results) in enumerate([[conf_labels, [conf_conf, sup_conf, norm_conf]], [sup_labels, [conf_sup, sup_sup, norm_sup]], [norm_labels, [conf_no, sup_no, norm_no]]]):
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
plt.savefig(f'roc_curves_coco_{split}.png', bbox_inches='tight')

