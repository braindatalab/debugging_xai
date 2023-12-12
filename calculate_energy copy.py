import os
import sys
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image 

SEED=1234
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap  
from captum.attr import GuidedBackprop, Deconvolution, LRP, InputXGradient, Lime

model_ind = sys.argv[1]
print(f'MODEL IND: {model_ind}')

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


def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val


with open('mark_all128.pkl', 'rb') as f:
    watermark_dataset = pickle.load(f)
    watermark_dataset = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in watermark_dataset]
print('test')

with open('no_mark_test128.pkl', 'rb') as f:
    no_watermark_dataset = pickle.load(f)
    no_watermark_dataset = [[rescale_values(i[0],1,0).transpose(2,0,1),i[1]] for i in no_watermark_dataset]

folder=os.getcwd()
watermark_path=folder+'/watermark banner.jpg'
watermark = Image.open(watermark_path)
w=int(watermark_dataset[0][0].shape[1])
h=int(watermark.size[1]*w/watermark.size[0])
watermark = watermark.resize((w,h))
watermark=np.array(watermark)
rgb=rescale_values(watermark,1,0)
r, g, blue = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
gray = 1-(0.2989 * r + 0.5870 * g + 0.1140 * blue)
white=np.ones((w,w))
white[0:gray.shape[0],0:gray.shape[1]]=gray

bin_water=1.0*(white<1)



def plot_atts(data,model,target):
    # data is a tensor of shape torch.Size([1, 3, 128, 128])
    # model is
    # target is an integer
    
    torch.manual_seed(SEED)
    out=model(data)
    Y_probs = F.softmax(out[0], dim=-1)
    
    ig_att = np.transpose(IntegratedGradients(model).attribute(data, target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    sal_att = np.transpose(Saliency(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    gradshap_att = np.transpose(GradientShap(model).attribute(data,target=target, baselines=torch.zeros(data.shape).to(DEVICE)).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    backprop_att = np.transpose(GuidedBackprop(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    ix_att = np.transpose(InputXGradient(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    deconv_att = np.transpose(Deconvolution(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    lrp_att=np.transpose(LRP(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    lime_att=np.transpose(Lime(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    grayscale_att_deconv = np.dot(deconv_att[...,:3], rgb_weights)
    grayscale_att_ix = np.dot(ix_att[...,:3], rgb_weights)
    grayscale_att_backprp = np.dot(backprop_att[...,:3], rgb_weights)
    grayscale_att_shap = np.dot(gradshap_att[...,:3], rgb_weights)
    grayscale_att_sal = np.dot(sal_att[...,:3], rgb_weights)
    grayscale_att_ig = np.dot(ig_att[...,:3], rgb_weights)
    grayscale_att_lrp = np.dot(lrp_att[...,:3], rgb_weights)
    grayscale_att_lime = np.dot(lime_att[...,:3], rgb_weights)
    
    
    atts={'deconv':abs(grayscale_att_deconv),'saliency':abs(grayscale_att_sal),'int_grads':abs(grayscale_att_ig),
          'shap':abs(grayscale_att_shap),'backprop':abs(grayscale_att_backprp),'ix':abs(grayscale_att_ix),
          'lrp':abs(grayscale_att_lrp), 'lime': abs(grayscale_att_lime)}
    
    return atts,Y_probs


def energy(att):
    image_size=att.shape[0]*att.shape[1]
    watermark_size=np.sum(bin_water)
    
    watermark_att=att*bin_water
    watermark_energy=np.sum(watermark_att)
    
    image_energy=np.sum(att)

    energy=(watermark_energy/watermark_size)/(image_energy/image_size)
    
    return energy

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    return model


energy_water_conf={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lime':[]}
energy_water_sup={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lime':[]}
energy_water_no={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lime':[]}

model_conf=load_trained(f'./cnn_conf_{model_ind}.pt').to(DEVICE)
model_sup=load_trained(f'./cnn_sup_{model_ind}.pt').to(DEVICE)
model_no=load_trained(f'./cnn_no_{model_ind}.pt').to(DEVICE)

folder=os.getcwd()+'/images'
print(folder)
t0=time.time()

energy_no_water_conf={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lime':[]}
energy_no_water_sup={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lime':[]}
energy_no_water_no={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lime':[]}
for i,w_image in enumerate(watermark_dataset):
    image=w_image[0]
    target=w_image[1]
    
    example=torch.tensor(image).unsqueeze(0).to(DEVICE,dtype=torch.float)
    
    a_conf_w,_=plot_atts(example,model_conf,target)
    a_sup_w,_=plot_atts(example,model_sup,target)
    a_no_w,_=plot_atts(example,model_no,target)

    image_nw=no_watermark_dataset[i][0]
    target_nw=no_watermark_dataset[i][1]
    
    example_nw=torch.tensor(image_nw).unsqueeze(0).to(DEVICE,dtype=torch.float)

    a_conf_nw,_=plot_atts(example_nw,model_conf,target)
    a_sup_nw,_=plot_atts(example_nw,model_sup,target)
    a_no_nw,_=plot_atts(example_nw,model_no,target)
    
    for method in list(energy_water_conf.keys()):
        energy_water_conf[method].append(energy(a_conf_w[method]))
        energy_water_sup[method].append(energy(a_sup_w[method]))
        energy_water_no[method].append(energy(a_no_w[method]))

        energy_no_water_conf[method].append(energy(a_conf_nw[method]))
        energy_no_water_sup[method].append(energy(a_sup_nw[method]))
        energy_no_water_no[method].append(energy(a_no_nw[method]))

    if (i%100)==0:
        print(i, 'out of', len(watermark_dataset))
        print(time.time()-t0)
        t0=time.time()


with open(f'energy_water_conf_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_conf, f)    
with open(f'energy_water_sup_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_sup, f)    
with open(f'energy_water_no_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_no, f)    
    
with open(f'energy_no_water_conf_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_conf, f)    
with open(f'energy_no_water_sup_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_sup, f)    
with open(f'energy_no_water_no_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_no, f)