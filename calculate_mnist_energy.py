import os

# os.environ["OPENBLAS_NUM_THREADS"] = "8"

# 0-15
# 16-31
# 32-47
# 48-63



import sys
import numpy as np
import time
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image 

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304]
SEED=SEEDS[int(sys.argv[1])]

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device("cuda",int(sys.argv[2]))

from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap  
from captum.attr import GuidedBackprop, Deconvolution, LRP, InputXGradient, Lime

from zennit.composites import EpsilonAlpha2Beta1

model_ind = sys.argv[1]
print(f'MODEL IND: {model_ind}')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

# proportion of channel attribution sum over total image attribution
def channel_sum(attr, channel):
    return np.sum(np.abs(attr[channel]))/np.sum(np.abs(attr))

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

        grad = torch.eye(10).to(DEVICE)[[target]].to(DEVICE)
        # gradient/ relevance wrt. class/output 0
        output.backward(gradient=grad.reshape((1,10)))
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


# Confounder model tested on conf/sup/no data
energy_red_conf_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_conf_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_conf_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

energy_red_conf_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_conf_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_conf_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

energy_red_conf_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_conf_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_conf_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# Suppressor model tested on conf/sup/no data
energy_red_sup_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_sup_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_sup_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

energy_red_sup_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_sup_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_sup_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

energy_red_sup_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_sup_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_sup_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

# No Colour model tested on conf/sup/no data
energy_red_no_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_no_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_no_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

energy_red_no_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_no_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_no_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

energy_red_no_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_green_no_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_blue_no_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}


energies = {
    'conf': { # conf model on conf/sup/no data
        'conf': [energy_red_conf_conf, energy_green_conf_conf, energy_blue_conf_conf],
        'sup': [energy_red_conf_sup, energy_green_conf_sup, energy_blue_conf_sup], 
        'no_col': [energy_red_conf_no, energy_green_conf_no, energy_blue_conf_no]},
            
    'sup': { # sup model on conf/sup/no data
        'conf': [energy_red_sup_conf, energy_green_sup_conf, energy_blue_sup_conf],
        'sup': [energy_red_sup_sup, energy_green_sup_sup, energy_blue_sup_sup], 
        'no_col': [energy_red_sup_no, energy_green_sup_no, energy_blue_sup_no]},

    'no_col': { # no_col model on conf/sup/no data
        'conf': [energy_red_no_conf, energy_green_no_conf, energy_blue_no_conf],
        'sup': [energy_red_no_sup, energy_green_no_sup, energy_blue_no_sup],
        'no_col': [energy_red_no_no, energy_green_no_no, energy_blue_no_no]}
}


# energies = {
#     'conf':[energy_red_conf, energy_green_conf, energy_blue_conf], 
#     'sup': [energy_red_sup, energy_green_sup, energy_blue_sup],
#     'no_col': [energy_red_no, energy_green_no, energy_blue_no]
#     }

# attrs = {
#     'conf': {'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []},
#     'sup': {'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []},
#     'no_col': {'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# }

with open('mnist_conf_data.pkl', 'rb') as f:
    conf_data = pkl.load(f)
    
with open('mnist_sup_data.pkl', 'rb') as f:
    sup_data = pkl.load(f)
    
with open('mnist_no_col_data.pkl', 'rb') as f:
    no_col_data = pkl.load(f)

batch_size = 64

confounder_test_loader = DataLoader(TensorDataset(torch.tensor(conf_data[2][0].transpose(0,3,1,2)), torch.tensor(conf_data[2][1])),batch_size=batch_size, shuffle=False)
suppressor_test_loader = DataLoader(TensorDataset(torch.tensor(sup_data[2][0].transpose(0,3,1,2)), torch.tensor(sup_data[2][1])),batch_size=batch_size, shuffle=False)
no_col_test_loader = DataLoader(TensorDataset(torch.tensor(no_col_data[2][0].transpose(0,3,1,2)), torch.tensor(no_col_data[2][1])),batch_size=batch_size, shuffle=False)

# model_conf=load_trained(f'./mnist_conf_{model_ind}.pt').to(DEVICE)
# model_sup=load_trained(f'./mnist_sup_{model_ind}.pt').to(DEVICE)
# model_no=load_trained(f'./mnist_no_col_{model_ind}.pt').to(DEVICE)

folder=os.getcwd()+'/images'
print(folder)
t0=time.time()


models = ['conf', 'sup', 'no_col']
atts = []
xs = []

for model_name, model_energies in energies.items():
    model = load_trained(f'./mnist_{model_name}_{model_ind}.pt').to(DEVICE)
    for i, test_loader in enumerate([confounder_test_loader, suppressor_test_loader, no_col_test_loader]):
        atts_loader = []
        xs_loader = []
        print('calculating attributions for model', model_name, 'with test dataset', models[i])
        for x_batch, test_labels in test_loader:      
            # outputs = model(x_test.to(torch.float))

            for j in range(x_batch.shape[0]):
                data = x_batch[j].reshape(1,3,28,28).to(torch.float32).to(DEVICE)
                target = torch.tensor(test_labels[j]).to(torch.int16).to(DEVICE)

                ig_att = IntegratedGradients(model).attribute(data, target=target).cpu().detach().numpy().squeeze()
                gradshap_att = GradientShap(model).attribute(data,target=target, baselines=torch.zeros(data.shape).to(DEVICE)).cpu().detach().numpy().squeeze()
                deconv_att = Deconvolution(model).attribute(data,target=target).cpu().detach().numpy().squeeze()
                lrp_att=LRP(model).attribute(data,target=target).cpu().detach().numpy().squeeze()
                lrp_ab = lrp(data,model,target)

                # attrs[models[i]]['deconv'].append(deconv_att)
                # attrs[models[i]]['int_grads'].append(ig_att)
                # attrs[models[i]]['shap'].append(gradshap_att)
                # attrs[models[i]]['lrp'].append(lrp_att)
                # attrs[models[i]]['lrp_ab'].append(lrp_ab)

                for channel_ind in range(3):
                    model_energies[models[i]][channel_ind]['deconv'].append(channel_sum(deconv_att, channel_ind))
                    model_energies[models[i]][channel_ind]['int_grads'].append(channel_sum(ig_att, channel_ind))
                    model_energies[models[i]][channel_ind]['shap'].append(channel_sum(gradshap_att, channel_ind))
                    model_energies[models[i]][channel_ind]['lrp'].append(channel_sum(lrp_att, channel_ind))
                    model_energies[models[i]][channel_ind]['lrp_ab'].append(channel_sum(lrp_ab, channel_ind))
                

# with open(f'attrs_mnist_{model_ind}.pickle', 'wb') as f:
#     pkl.dump(attrs, f)

with open(f'energies_mnist_{model_ind}.pickle', 'wb') as f:
    pkl.dump(energies, f)


# for i in range(len(watermark_dataset)):
#     w_image = watermark_dataset[i]
#     w_image=w_image[0]
#     w_target=watermark_dataset[i][1]
#     w_example=torch.tensor(w_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

#     w_target_conf = torch.max(model_conf(w_example), 1)[1].to(int)
#     w_target_sup = torch.max(model_sup(w_example), 1)[1].to(int)
#     w_target_no = torch.max(model_no(w_example), 1)[1].to(int)

#     nw_image = no_watermark_dataset[i]
#     nw_image=nw_image[0]
#     nw_target=watermark_dataset[i][1]
#     nw_example=torch.tensor(nw_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

#     nw_target_conf = torch.max(model_conf(nw_example), 1)[1].to(int)
#     nw_target_sup = torch.max(model_sup(nw_example), 1)[1].to(int)
#     nw_target_no = torch.max(model_no(nw_example), 1)[1].to(int)

#     # explanations for predicted class (n)w_target_{MODEL}
#     a_conf_w,_=plot_atts(w_example,model_conf,w_target_conf)
#     a_sup_w,_=plot_atts(w_example,model_sup,w_target_sup)
#     a_no_w,_=plot_atts(w_example,model_no,w_target_no)

#     a_conf_nw,_=plot_atts(nw_example,model_conf,nw_target_conf)
#     a_sup_nw,_=plot_atts(nw_example,model_sup,nw_target_sup)
#     a_no_nw,_=plot_atts(nw_example,model_no,nw_target_no)
    
#     # explanations for ground truth class (n)w_target
#     # a_conf_w_gt,_=plot_atts(w_example,model_conf,w_target)
#     # a_sup_w_gt,_=plot_atts(w_example,model_sup,w_target)
#     # a_no_w_gt,_=plot_atts(w_example,model_no,w_target)

#     # a_conf_nw_gt,_=plot_atts(nw_example,model_conf,nw_target)
#     # a_sup_nw_gt,_=plot_atts(nw_example,model_sup,nw_target)
#     # a_no_nw_gt,_=plot_atts(nw_example,model_no,nw_target)

#     for method in list(energy_water_conf.keys()):
#         energy_water_conf[method].append(energy(a_conf_w[method]))
#         energy_water_sup[method].append(energy(a_sup_w[method]))
#         energy_water_no[method].append(energy(a_no_w[method]))

#         energy_no_water_conf[method].append(energy(a_conf_nw[method]))
#         energy_no_water_sup[method].append(energy(a_sup_nw[method]))
#         energy_no_water_no[method].append(energy(a_no_nw[method]))

#         # energy_water_conf_gt[method].append(energy(a_conf_w_gt[method]))
#         # energy_water_sup_gt[method].append(energy(a_sup_w_gt[method]))
#         # energy_water_no_gt[method].append(energy(a_no_w_gt[method]))

#         # energy_no_water_conf_gt[method].append(energy(a_conf_nw_gt[method]))
#         # energy_no_water_sup_gt[method].append(energy(a_sup_nw_gt[method]))
#         # energy_no_water_no_gt[method].append(energy(a_no_nw_gt[method]))

#     if (i%100)==0:
#         print(i, 'out of', len(watermark_dataset))
#         print(time.time()-t0)
#         # t0=time.time()


# with open(f'energy_water_conf_comb_pred_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_conf, f)    
# with open(f'energy_water_sup_comb_pred_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_sup, f)    
# with open(f'energy_water_no_comb_pred_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_no, f)    
    
# with open(f'energy_no_water_conf_comb_pred_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_conf, f)    
# with open(f'energy_no_water_sup_comb_pred_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_sup, f)    
# with open(f'energy_no_water_no_comb_pred_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_no, f)


# # with open(f'energy_water_conf_gt_{model_ind}.pickle', 'wb') as f:
# #     pickle.dump(energy_water_conf_gt, f)    
# # with open(f'energy_water_sup_gt_{model_ind}.pickle', 'wb') as f:
# #     pickle.dump(energy_water_sup_gt, f)    
# # with open(f'energy_water_no_gt_{model_ind}.pickle', 'wb') as f:
# #     pickle.dump(energy_water_no_gt, f)    
    
# # with open(f'energy_no_water_conf_gt_{model_ind}.pickle', 'wb') as f:
# #     pickle.dump(energy_no_water_conf_gt, f)    
# # with open(f'energy_no_water_sup_gt_{model_ind}.pickle', 'wb') as f:
# #     pickle.dump(energy_no_water_sup_gt, f)    
# # with open(f'energy_no_water_no_gt_{model_ind}.pickle', 'wb') as f:
# #     pickle.dump(energy_no_water_no_gt, f)
