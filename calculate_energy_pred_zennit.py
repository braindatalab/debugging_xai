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

if sys.argv[2] is not None:
    DEVICE = torch.device("cuda",int(sys.argv[2]))
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = 'cpu'

# from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap  
# from captum.attr import GuidedBackprop, Deconvolution, LRP, InputXGradient, Lime

from zennit.composites import EpsilonAlpha2Beta1
# https://zennit.readthedocs.io/en/latest/how-to/use-rules-composites-and-canonizers.html#abstract-composites

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

def lrp(data,model,target,device):
    # create a composite instance
    #composite = EpsilonPlusFlat()
    composite = EpsilonAlpha2Beta1()

    # use the following instead to ignore bias for the relevance
    # composite = EpsilonPlusFlat(zero_params='bias')

    # make sure the input requires a gradient
    data.requires_grad = True

    # compute the output and gradient within the composite's context
    with composite.context(model) as modified_model:
        modified_model=modified_model.to(device)
        output = modified_model(data.to(device)).to(device)

        grad = torch.eye(2)[[target]].to(device)
        # gradient/ relevance wrt. class/output 0
        output.backward(gradient=grad)
        # relevance is not accumulated in .grad if using torch.autograd.grad
        # relevance, = torch.autograd.grad(output, input, torch.eye(10)[[0])

    # gradient is accumulated in input.grad
    att=abs(data.grad.detach().cpu().squeeze().numpy().transpose(1,2,0))

    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_att_lrp = np.dot(att[...,:3], rgb_weights)

    
    return grayscale_att_lrp


# def energy(att):
#     image_size=att.shape[0]*att.shape[1]
#     watermark_size=np.sum(bin_water)
    
#     watermark_att=att*bin_water
#     watermark_energy=np.sum(watermark_att)
    
#     image_energy=np.sum(att)

#     energy=(watermark_energy/watermark_size)/(image_energy/image_size)
    
#     return energy

def energy(att):
    image_size=att.shape[0]*att.shape[1]
    watermark_size=np.sum(bin_water)
    
    watermark_att=att*bin_water
    watermark_energy=np.sum(watermark_att)
    
    image_energy=np.sum(att)

    
    # Gives energy(watermark) = 7.858
#     energy=(watermark_energy/watermark_size)/(image_energy/image_size)
    
    # Gives energy(watermark) = 1.0
    energy = (np.sum(att*bin_water)/att.shape[0]*att.shape[1]) / (np.sum(att)/att.shape[0]*att.shape[1])
    
    return energy

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    return model


energy_water_conf={'lrp':[]}
energy_water_sup={'lrp':[]}
energy_water_no={'lrp':[]}

energy_no_water_conf={'lrp':[]}
energy_no_water_sup={'lrp':[]}
energy_no_water_no={'lrp':[]}

model_conf=load_trained(f'./cnn_conf_{model_ind}.pt').to(DEVICE)
model_sup=load_trained(f'./cnn_sup_{model_ind}.pt').to(DEVICE)
model_no=load_trained(f'./cnn_no_{model_ind}.pt').to(DEVICE)

folder=os.getcwd()+'/images'
print(folder)
t0=time.time()


for i in range(len(watermark_dataset)):
    w_image = watermark_dataset[i]
    w_image=w_image[0]
    # w_target=w_image[1]
    w_example=torch.tensor(w_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

    w_target_conf = torch.max(model_conf(w_example), 1)[1].to(int)
    w_target_sup = torch.max(model_sup(w_example), 1)[1].to(int)
    w_target_no = torch.max(model_no(w_example), 1)[1].to(int)

    nw_image = no_watermark_dataset[i]
    nw_image=nw_image[0]
    # nw_target=w_image[1]
    nw_example=torch.tensor(nw_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

    nw_target_conf = torch.max(model_conf(nw_example), 1)[1].to(int)
    nw_target_sup = torch.max(model_sup(nw_example), 1)[1].to(int)
    nw_target_no = torch.max(model_no(nw_example), 1)[1].to(int)
    
    a_conf_w =lrp(w_example,model_conf,w_target_conf, DEVICE)
    a_sup_w =lrp(w_example,model_sup,w_target_sup, DEVICE)
    a_no_w =lrp(w_example,model_no,w_target_no, DEVICE)

    a_conf_nw =lrp(nw_example,model_conf,nw_target_conf, DEVICE)
    a_sup_nw =lrp(nw_example,model_sup,nw_target_sup, DEVICE)
    a_no_nw =lrp(nw_example,model_no,nw_target_no, DEVICE)
    
    # for method in list(energy_water_conf.keys()):
    energy_water_conf['lrp'].append(energy(a_conf_w))
    energy_water_sup['lrp'].append(energy(a_sup_w))
    energy_water_no['lrp'].append(energy(a_no_w))

    energy_no_water_conf['lrp'].append(energy(a_conf_nw))
    energy_no_water_sup['lrp'].append(energy(a_sup_nw))
    energy_no_water_no['lrp'].append(energy(a_no_nw))

    if (i%100)==0:
        print(i, 'out of', len(watermark_dataset))
        print(time.time()-t0)
        # t0=time.time()


with open(f'zennit_energy_water_conf_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_conf, f)    
with open(f'zennit_energy_water_sup_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_sup, f)    
with open(f'zennit_energy_water_no_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_no, f)    
    
with open(f'zennit_energy_no_water_conf_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_conf, f)    
with open(f'zennit_energy_no_water_sup_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_sup, f)    
with open(f'zennit_energy_no_water_no_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_no, f)
