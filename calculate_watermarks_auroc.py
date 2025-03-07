import torch
from torch import manual_seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy,AUROC
import sys

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
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*128*8, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 1028),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(1028, num_classes))

    def forward(self, x):
        out = self.layer1(x)

        out = self.layer3(out)

        out = self.layer5(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out        

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    return model


def print_AUC(loader, model_ind=0, split=0):

    cnn_conf=load_trained(f'./models/cnn_confounder_{split}_{model_ind}.pt').eval().to(DEVICE)
    cnn_sup=load_trained(f'./models/cnn_suppressor_{split}_{model_ind}.pt').eval().to(DEVICE)
    cnn_no=load_trained(f'./models/cnn_no_watermark_{split}_{model_ind}.pt').eval().to(DEVICE)
    
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

for split in [sys.argv[1]]:
    for i in range(5):
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

        with open(f'./artifacts/split_{split}_confounder_test.pkl', 'rb') as f:
            confounder_test, labels_test_conf, _ = pickle.load(f)
            confounder_test = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_conf.flatten()[i]] for i,x in enumerate(confounder_test)]

        with open(f'./artifacts/split_{split}_suppressor_test.pkl', 'rb') as f:
            suppressor_test, labels_test_sup, _ = pickle.load(f)
            suppressor_test = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_sup.flatten()[i]] for i,x in enumerate(suppressor_test)]

        with open(f'./artifacts/split_{split}_no_watermark_test.pkl', 'rb') as f:
            no_mark_test, labels_test_no, _ = pickle.load(f)
            no_mark_test = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_no.flatten()[i]] for i,x in enumerate(no_mark_test)]

        confounder_test_loader = DataLoader(confounder_test,batch_size=batch_size, shuffle=True)
        suppressor_test_loader = DataLoader(suppressor_test,batch_size=batch_size, shuffle=True)
        no_mark_test_loader = DataLoader(no_mark_test,batch_size=batch_size, shuffle=True)

        conf_scores, [conf_model_conf_test, sup_model_conf_test, no_model_conf_test, conf_true_labels] = print_AUC(confounder_test_loader, i, split)

        print()
        print('suppressor data:')
        sup_scores, [conf_model_sup_test, sup_model_sup_test, no_model_sup_test, sup_true_labels] = print_AUC(suppressor_test_loader, i, split)

        print()
        print('no watermark data:')
        no_scores, [conf_model_no_test, sup_model_no_test, no_model_no_test, no_true_labels] = print_AUC(no_mark_test_loader, i, split)

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

print('                        MODEL   ')
print('MEAN AUROC            CONF    SUP    NO')
print('conf data: ', torch.mean(torch.tensor(confounder_data_results), axis=0))
print('sup data: ', torch.mean(torch.tensor(suppressor_data_results), axis=0))
print('no mark data: ', torch.mean(torch.tensor(no_mark_data_results), axis=0))

print('STD AUROC')
print('conf data: ', torch.std(torch.tensor(confounder_data_results), axis=0))
print('sup data: ', torch.std(torch.tensor(suppressor_data_results), axis=0))
print('no mark data: ', torch.std(torch.tensor(no_mark_data_results), axis=0))

# for split in range(5):
#     conf_results = []
#     sup_results = []
#     no_results = []

#     for i in range(5):
#         conf_results.append(confounder_data_results[split*5+i])
#         sup_results.append(suppressor_data_results[split*5+i])
#         no_results.append(no_mark_data_results[split*5+i])
    
#     print(f'           SPLIT {split}       ')
#     print('                        MODEL   ')
#     print('AUROC                 CONF    SUP    NO')
#     print('conf data: ', torch.mean(torch.tensor(conf_results),axis=0))
#     print('sup data: ', torch.mean(torch.tensor(sup_results),axis=0))
#     print('no mark data: ', torch.mean(torch.tensor(no_results),axis=0))

#     print('STD')
#     print('conf data', torch.std(torch.tensor(conf_results),axis=0))
#     print('sup data', torch.std(torch.tensor(sup_results),axis=0))
#     print('no mark data', torch.std(torch.tensor(no_results),axis=0))


import pickle as pkl
split = sys.argv[1]

with open(f'./auroc_results_split_{split}.pickle', 'wb') as f:
    pickle.dump([confounder_data_results, suppressor_data_results, no_mark_data_results], f)

with open(f'./auroc_conf_{split}.pickle', 'wb') as f:
    pickle.dump([conf_conf, conf_sup, conf_no], f)    

with open(f'./auroc_sup_{split}.pickle', 'wb') as f:
    pickle.dump([sup_conf, sup_sup, sup_no], f)

with open(f'./auroc_no_{split}.pickle', 'wb') as f:
    pickle.dump([no_conf, no_sup, no_no], f)

with open(f'./conf_labels_{split}.pickle', 'wb') as f:
    pickle.dump(conf_labels, f)

with open(f'./sup_labels_{split}.pickle', 'wb') as f:
    pickle.dump(sup_labels, f)

with open(f'./no_labels_{split}.pickle', 'wb') as f:
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
plt.savefig(f'./figures/roc_curves_{split}.png', bbox_inches='tight')

