import os

# os.environ["OPENBLAS_NUM_THREADS"] = "8"

# 0-15
# 16-31
# 32-47
# 48-63



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

# if sys.argv[2] is not None:
#     DEVICE = torch.device("cuda",int(sys.argv[2]))
# else:
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device("cuda",int(sys.argv[2]))
# DEVICE = 'cpu'

from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap  
from captum.attr import GuidedBackprop, Deconvolution, LRP, InputXGradient, Lime

from zennit.composites import EpsilonAlpha2Beta1

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
        output.backward(gradient=grad)
        # relevance is not accumulated in .grad if using torch.autograd.grad
        # relevance, = torch.autograd.grad(output, input, torch.eye(10)[[0])

    # gradient is accumulated in input.grad
    att=abs(data.grad.detach().cpu().squeeze().numpy().transpose(1,2,0))

    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_att_lrp = np.dot(att[...,:3], rgb_weights)

    
    return grayscale_att_lrp

def plot_atts(data,model,target):
    # data is a tensor of shape torch.Size([1, 3, 128, 128])
    # model is
    # target is an integer
    
    torch.manual_seed(SEED)
    out=model(data)
    Y_probs = F.softmax(out[0], dim=-1)
    
    ig_att = np.transpose(IntegratedGradients(model).attribute(data, target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # sal_att = np.transpose(Saliency(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    gradshap_att = np.transpose(GradientShap(model).attribute(data,target=target, baselines=torch.zeros(data.shape).to(DEVICE)).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # backprop_att = np.transpose(GuidedBackprop(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # ix_att = np.transpose(InputXGradient(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    deconv_att = np.transpose(Deconvolution(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    lrp_att=np.transpose(LRP(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # lime_att=np.transpose(Lime(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    
    lrp_ab = lrp(data,model,target)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    grayscale_att_deconv = np.dot(deconv_att[...,:3], rgb_weights)
    # grayscale_att_ix = np.dot(ix_att[...,:3], rgb_weights)
    # grayscale_att_backprp = np.dot(backprop_att[...,:3], rgb_weights)
    grayscale_att_shap = np.dot(gradshap_att[...,:3], rgb_weights)
    # grayscale_att_sal = np.dot(sal_att[...,:3], rgb_weights)
    grayscale_att_ig = np.dot(ig_att[...,:3], rgb_weights)
    grayscale_att_lrp = np.dot(lrp_att[...,:3], rgb_weights)
    # grayscale_att_lime = np.dot(lime_att[...,:3], rgb_weights)
    
    
    atts={'deconv':abs(grayscale_att_deconv),
        #   'saliency':abs(grayscale_att_sal),
          'int_grads':abs(grayscale_att_ig),
          'shap':abs(grayscale_att_shap),
        #   'backprop':abs(grayscale_att_backprp),
        #   'ix':abs(grayscale_att_ix),
          'lrp':abs(grayscale_att_lrp), 
          'lrp_ab':abs(lrp_ab)
        #   'lime': abs(grayscale_att_lime)
          }
    
    return atts,Y_probs


def energy(att):
    image_size=att.shape[0]*att.shape[1]
    watermark_size=np.sum(bin_water)
    
    watermark_att=att*bin_water
    watermark_energy=np.sum(watermark_att)
    
    image_energy=np.sum(att)

    energy=(watermark_energy/watermark_size)/(image_energy/image_size)
    
    return energy

# def energy(att):
#     image_size=att.shape[0]*att.shape[1]
#     watermark_size=np.sum(bin_water)
    
#     watermark_att=att*bin_water
#     watermark_energy=np.sum(watermark_att)
    
#     image_energy=np.sum(att)

    
#     # Gives energy(watermark) = 7.858
#     energy=(watermark_energy/watermark_size)/(image_energy/image_size)
    
#     # Gives energy(watermark) = 1.0
#     # energy = (np.sum(att*bin_water)/att.shape[0]*att.shape[1]) / (np.sum(att)/att.shape[0]*att.shape[1])
    
#     return energy

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    return model


energy_water_conf={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
energy_water_sup={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
energy_water_no={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}

energy_no_water_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_no_water_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_no_water_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}


# energy_water_conf_gt={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
# energy_water_sup_gt={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
# energy_water_no_gt={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}

# energy_no_water_conf_gt={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_no_water_sup_gt={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
# energy_no_water_no_gt={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}

model_conf=load_trained(f'./cnn_conf_{model_ind}.pt').to(DEVICE)
model_sup=load_trained(f'./cnn_sup_{model_ind}.pt').to(DEVICE)
model_no=load_trained(f'./cnn_no_{model_ind}.pt').to(DEVICE)

folder=os.getcwd()+'/images'
print(folder)
t0=time.time()


for i in range(len(watermark_dataset)):
    w_image = watermark_dataset[i]
    w_image=w_image[0]
    w_target=watermark_dataset[i][1]
    w_example=torch.tensor(w_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

    w_target_conf = torch.max(model_conf(w_example), 1)[1].to(int)
    w_target_sup = torch.max(model_sup(w_example), 1)[1].to(int)
    w_target_no = torch.max(model_no(w_example), 1)[1].to(int)

    nw_image = no_watermark_dataset[i]
    nw_image=nw_image[0]
    nw_target=watermark_dataset[i][1]
    nw_example=torch.tensor(nw_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

    nw_target_conf = torch.max(model_conf(nw_example), 1)[1].to(int)
    nw_target_sup = torch.max(model_sup(nw_example), 1)[1].to(int)
    nw_target_no = torch.max(model_no(nw_example), 1)[1].to(int)

    # explanations for predicted class (n)w_target_{MODEL}
    a_conf_w,_=plot_atts(w_example,model_conf,w_target_conf)
    a_sup_w,_=plot_atts(w_example,model_sup,w_target_sup)
    a_no_w,_=plot_atts(w_example,model_no,w_target_no)

    a_conf_nw,_=plot_atts(nw_example,model_conf,nw_target_conf)
    a_sup_nw,_=plot_atts(nw_example,model_sup,nw_target_sup)
    a_no_nw,_=plot_atts(nw_example,model_no,nw_target_no)
    
    # explanations for ground truth class (n)w_target
    # a_conf_w_gt,_=plot_atts(w_example,model_conf,w_target)
    # a_sup_w_gt,_=plot_atts(w_example,model_sup,w_target)
    # a_no_w_gt,_=plot_atts(w_example,model_no,w_target)

    # a_conf_nw_gt,_=plot_atts(nw_example,model_conf,nw_target)
    # a_sup_nw_gt,_=plot_atts(nw_example,model_sup,nw_target)
    # a_no_nw_gt,_=plot_atts(nw_example,model_no,nw_target)

    for method in list(energy_water_conf.keys()):
        energy_water_conf[method].append(energy(a_conf_w[method]))
        energy_water_sup[method].append(energy(a_sup_w[method]))
        energy_water_no[method].append(energy(a_no_w[method]))

        energy_no_water_conf[method].append(energy(a_conf_nw[method]))
        energy_no_water_sup[method].append(energy(a_sup_nw[method]))
        energy_no_water_no[method].append(energy(a_no_nw[method]))

        # energy_water_conf_gt[method].append(energy(a_conf_w_gt[method]))
        # energy_water_sup_gt[method].append(energy(a_sup_w_gt[method]))
        # energy_water_no_gt[method].append(energy(a_no_w_gt[method]))

        # energy_no_water_conf_gt[method].append(energy(a_conf_nw_gt[method]))
        # energy_no_water_sup_gt[method].append(energy(a_sup_nw_gt[method]))
        # energy_no_water_no_gt[method].append(energy(a_no_nw_gt[method]))

    if (i%100)==0:
        print(i, 'out of', len(watermark_dataset))
        print(time.time()-t0)
        # t0=time.time()


with open(f'energy_water_conf_comb_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_conf, f)    
with open(f'energy_water_sup_comb_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_sup, f)    
with open(f'energy_water_no_comb_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_no, f)    
    
with open(f'energy_no_water_conf_comb_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_conf, f)    
with open(f'energy_no_water_sup_comb_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_sup, f)    
with open(f'energy_no_water_no_comb_pred_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_no, f)


# with open(f'energy_water_conf_gt_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_conf_gt, f)    
# with open(f'energy_water_sup_gt_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_sup_gt, f)    
# with open(f'energy_water_no_gt_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_no_gt, f)    
    
# with open(f'energy_no_water_conf_gt_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_conf_gt, f)    
# with open(f'energy_no_water_sup_gt_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_sup_gt, f)    
# with open(f'energy_no_water_no_gt_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_no_gt, f)