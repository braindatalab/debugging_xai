import os
import sys
import numpy as np
import time
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image 
import cv2

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

SEED=SEEDS[1] 

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device("cuda",int(sys.argv[1]))

from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap  
from captum.attr import GuidedBackprop, Deconvolution, LRP, InputXGradient, Lime

from zennit.composites import EpsilonAlpha2Beta1

split = sys.argv[1]
model_ind = int(sys.argv[2])

print(f'SPLIT: {split}; MODEL IND: {model_ind}')

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


def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

# proportion of channel attribution sum over total image attribution
# def channel_sum(attr, channel):
#     return np.sum(np.abs(attr[channel]))/np.sum(np.abs(attr))

def channel_ratio(attr, channel):
    return np.sum(np.abs(attr[:,channel]))/(np.sum(np.abs(attr)) - np.sum(np.abs(attr[:,channel])))

def lrp(data,model,target):
    # create a composite instance
    #composite = EpsilonPlusFlat()
    composite = EpsilonAlpha2Beta1()

    # use the following instead to ignore bias for the relevance
    # composite = EpsilonPlusFlat(zero_params='bias')

    # make sure the input requires a gradient
    data.requires_grad = True

    # compute the output and gradient within the composite's context
    with composite.context(model) as modified_model:
        modified_model=modified_model.to(DEVICE)
        output = modified_model(data.to(DEVICE)).to(DEVICE)

        grad = torch.eye(2).to(DEVICE)[[target]].to(DEVICE)
        # gradient/ relevance wrt. class/output 0
        output.backward(gradient=grad.reshape((1,2)))
        # relevance is not accumulated in .grad if using torch.autograd.grad
        # relevance, = torch.autograd.grad(output, input, torch.eye(10)[[0])

    # gradient is accumulated in input.grad
    att=data.grad.detach().cpu().squeeze().numpy()

#    rgb_weights = [0.2989, 0.5870, 0.1140]
#    grayscale_att_lrp = np.dot(att[...,:3], rgb_weights)

    
    return att


def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    return model


# # Confounder model tested on conf/sup/no data
# energy_red_conf_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_conf_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_conf_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# energy_red_conf_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_conf_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_conf_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# energy_red_conf_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_conf_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_conf_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# # Suppressor model tested on conf/sup/no data
# energy_red_sup_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_sup_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_sup_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# energy_red_sup_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_sup_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_sup_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# energy_red_sup_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_sup_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_sup_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# # No Colour model tested on conf/sup/no data
# energy_red_no_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_no_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_no_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# energy_red_no_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_no_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_no_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# energy_red_no_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_green_no_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_blue_no_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}


# energies = {
#     'conf': { # conf model on conf/sup/no data
#         'conf': [energy_red_conf_conf, energy_green_conf_conf, energy_blue_conf_conf],
#         'sup': [energy_red_conf_sup, energy_green_conf_sup, energy_blue_conf_sup], 
#         'norm': [energy_red_conf_no, energy_green_conf_no, energy_blue_conf_no]},
            
#     'sup': { # sup model on conf/sup/no data
#         'conf': [energy_red_sup_conf, energy_green_sup_conf, energy_blue_sup_conf],
#         'sup': [energy_red_sup_sup, energy_green_sup_sup, energy_blue_sup_sup], 
#         'norm': [energy_red_sup_no, energy_green_sup_no, energy_blue_sup_no]},

#     'norm': { # no_col model on conf/sup/no data
#         'conf': [energy_red_no_conf, energy_green_no_conf, energy_blue_no_conf],
#         'sup': [energy_red_no_sup, energy_green_no_sup, energy_blue_no_sup],
#         'norm': [energy_red_no_no, energy_green_no_no, energy_blue_no_no]}
# }


energies = {}
energies_rgb = {}

for model in ['conf', 'sup', 'norm']:
    energies[model] = {}
    energies_rgb[model] = {}
    for dataset in ['conf', 'sup', 'norm']:
        energies[model][dataset] = []
        energies_rgb[model][dataset] = []
        for i in range(3):
            energies[model][dataset].append({'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []})
            energies_rgb[model][dataset].append({'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []})

batch_size = 64

# with open(f'coco_data_norm.pkl', 'rb') as f:
#     [(_, _, _), (_, _, _), (x_test, y_test, masks_test_norm)] = pkl.load(f)



with open(f'./artifacts/split_{split}_coco_data_norm_test.pkl', 'rb') as f:
    [x_test_norm, y_test_norm, masks_test_norm, _] = pkl.load(f)

norm_loader = DataLoader(TensorDataset(torch.tensor(x_test_norm.transpose(0,3,1,2)), torch.tensor(y_test_norm)), batch_size=batch_size, shuffle=False, num_workers=4)

with open(f'./artifacts/split_{split}_coco_data_conf_test.pkl', 'rb') as f:
    [x_test_conf, y_test_conf, masks_test_conf, _] = pkl.load(f)

# with open(f'coco_data_conf.pkl', 'rb') as f:
#     [(_, _, _), (_, _, _), (x_test, y_test, masks_test_conf)] = pkl.load(f)

conf_loader = DataLoader(TensorDataset(torch.tensor(x_test_conf.transpose(0,3,1,2)), torch.tensor(y_test_conf)), batch_size=batch_size, shuffle=False, num_workers=4)

with open(f'./artifacts/split_{split}_coco_data_sup_test.pkl', 'rb') as f:
    [x_test_sup, y_test_sup, masks_test_sup, _] = pkl.load(f)

sup_loader = DataLoader(TensorDataset(torch.tensor(x_test_sup.transpose(0,3,1,2)), torch.tensor(y_test_sup)), batch_size=batch_size, shuffle=False, num_workers=4)


folder=os.getcwd()+'/images'
print(folder)
t0=time.time()


models = ['conf', 'sup', 'norm']
atts = []
xs = []

# print(conf_loader[0][0].shape)

energies_x = []
energies_x_rgb = []


for model_name, model_energies in energies.items():
    # if model_name == 'norm':
    #     model = load_trained(f'./models/coco_{model_name}_{model_ind}.pt').eval().to(DEVICE)
    # else:
    model = load_trained(f'./models/coco_{model_name}_{split}_{model_ind}.pt').eval().to(DEVICE)
    for i, test_loader in enumerate([conf_loader, sup_loader, norm_loader]):
        atts_loader = []
        xs_loader = []
        print('calculating attributions for model', model_name, 'with test dataset', models[i])

        x_energies = [[], [], []]
        x_energies_rgb = [[], [], []]
        for x_batch, test_labels in test_loader:      

            for j in range(x_batch.shape[0]):
                
                data = x_batch[j].reshape(1,3,224,224).to(torch.float32).to(DEVICE)
                target = torch.tensor(test_labels[j]).to(torch.int16).to(DEVICE)

                ig_att = IntegratedGradients(model).attribute(data, target=target).cpu().detach().numpy().squeeze()
                gradshap_att = GradientShap(model).attribute(data,target=target, baselines=torch.zeros(data.shape).to(DEVICE)).cpu().detach().numpy().squeeze()
                deconv_att = Deconvolution(model).attribute(data,target=target).cpu().detach().numpy().squeeze()
                lrp_att=LRP(model).attribute(data,target=target).cpu().detach().numpy().squeeze()
                lrp_ab = lrp(data,model,target)

                ig_rgb = np.transpose(cv2.cvtColor( (np.transpose(ig_att, (1,2,0))*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ), (2,0,1))
                gradshap_rgb = np.transpose(cv2.cvtColor( (np.transpose(gradshap_att, (1,2,0))*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ), (2,0,1))
                deconv_rgb = np.transpose(cv2.cvtColor( (np.transpose(deconv_att, (1,2,0))*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ), (2,0,1))
                lrp_rgb = np.transpose(cv2.cvtColor( (np.transpose(lrp_att, (1,2,0))*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ), (2,0,1))
                lrp_ab_rgb = np.transpose(cv2.cvtColor( (np.transpose(lrp_ab, (1,2,0))*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ), (2,0,1))

                # only need to calculate x energies once
                if model_name == 'conf':
                    rgb = np.transpose(cv2.cvtColor( (np.transpose(data, (1,2,0))*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ), (2,0,1))

                for channel_ind in range(3):
                    # like above, only need to calculate x (data) energies once, as it is model independent
                    if model_name == 'conf':
                        x_energies[channel_ind].append(channel_ratio(data, channel_ind))
                        x_energies_rgb[channel_ind].append(channel_ratio(rgb, channel_ind))

                    model_energies[models[i]][channel_ind]['deconv'].append(channel_ratio(deconv_att, channel_ind))
                    model_energies[models[i]][channel_ind]['int_grads'].append(channel_ratio(ig_att, channel_ind))
                    model_energies[models[i]][channel_ind]['shap'].append(channel_ratio(gradshap_att, channel_ind))
                    model_energies[models[i]][channel_ind]['lrp'].append(channel_ratio(lrp_att, channel_ind))
                    model_energies[models[i]][channel_ind]['lrp_ab'].append(channel_ratio(lrp_ab, channel_ind))

                    energies_rgb[model_name][models[i]][channel_ind]['deconv'].append(channel_ratio(deconv_rgb, channel_ind))
                    energies_rgb[model_name][models[i]][channel_ind]['int_grads'].append(channel_ratio(ig_rgb, channel_ind))
                    energies_rgb[model_name][models[i]][channel_ind]['shap'].append(channel_ratio(gradshap_rgb, channel_ind))
                    energies_rgb[model_name][models[i]][channel_ind]['lrp'].append(channel_ratio(lrp_rgb, channel_ind))
                    energies_rgb[model_name][models[i]][channel_ind]['lrp_ab'].append(channel_ratio(lrp_ab_rgb, channel_ind))
        energies_x.append(x_energies)
        energies_x_rgb.append(x_energies_rgb)

with open(f'energies_coco_{split}_{model_ind}.pickle', 'wb') as f:
    pkl.dump(energies, f)

with open(f'energies_coco_{split}_{model_ind}_rgb.pickle', 'wb') as f:
    pkl.dump(energies_rgb, f)

    
with open(f'energies_coco_x_{split}.pickle', 'wb') as f:
    pkl.dump(energies_x, f)

with open(f'energies_coco_x_rgb_{split}.pickle', 'wb') as f:
    pkl.dump(energies_x_rgb, f)
