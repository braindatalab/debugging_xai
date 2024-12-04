import os
import numpy as np

import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random

from captum.attr import IntegratedGradients, Saliency, GradientShap  
from captum.attr import GuidedBackprop, Deconvolution, LRP, InputXGradient, Lime

from captum._utils.models.linear_model import SkLearnLasso

from zennit.composites import EpsilonAlpha2Beta1

from torch.utils.data import DataLoader, TensorDataset

# device='cuda:6'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

SEED=SEEDS[1] 

np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

def load_model(path):
    model = Net()
    
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    model.eval()
    model.zero_grad()
    
    model.relu=nn.ReLU(inplace=False)
    return model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5, padding='same')
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding='same')
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5, padding='same')
        
        self.maxPooling4_0=nn.MaxPool2d(kernel_size=4)
        self.maxPooling4_1=nn.MaxPool2d(kernel_size=4)
        self.maxPooling2=nn.MaxPool2d(kernel_size=2)
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

def plot_atts(data,model,target):
    # DEVICE = 'cpu'
    # data is a tensor of shape torch.Size([1, 3, 128, 128])
    # model is
    # target is an integer

    ig_att = IntegratedGradients(model).attribute(data, target=target).cpu().detach().numpy().squeeze()
    gradshap_att = GradientShap(model).attribute(data,target=target, baselines=torch.zeros(data.shape).to(DEVICE)).cpu().detach().numpy().squeeze()
    deconv_att = Deconvolution(model).attribute(data,target=target).cpu().detach().numpy().squeeze()
    lrp_att=LRP(model).attribute(data,target=target).cpu().detach().numpy().squeeze()
    lrp_ab = lrp(data,model,target)
    
    out=model(data)
    Y_probs = F.softmax(out[0], dim=-1)
    target = int(target)
    # with torch.no_grad():
    ig_att = np.transpose(IntegratedGradients(model).attribute(data, target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # sal_att = np.transpose(Saliency(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    gradshap_att = np.transpose(GradientShap(model).attribute(data,target=target, baselines=torch.zeros(data.shape).to(device)).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # backprop_att = np.transpose(GuidedBackprop(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # ix_att = np.transpose(InputXGradient(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    deconv_att = np.transpose(Deconvolution(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    lrp_att=np.transpose(LRP(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # lime_att=np.transpose(Lime(model,interpretable_model=SkLearnLasso(alpha=0.01)).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    
    lrp_ab = lrp(data, model, target, DEVICE)
    
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    grayscale_att_deconv = np.dot(deconv_att[...,:3], rgb_weights)
    # grayscale_att_ix = np.dot(ix_att[...,:3], rgb_weights)
    # grayscale_att_backprp = np.dot(backprop_att[...,:3], rgb_weights)
    grayscale_att_shap = np.dot(gradshap_att[...,:3], rgb_weights)
    # grayscale_att_sal = np.dot(sal_att[...,:3], rgb_weights)
    grayscale_att_ig = np.dot(ig_att[...,:3], rgb_weights)
    grayscale_att_lrp = np.dot(lrp_att[...,:3], rgb_weights)
    # grayscale_att_lime = np.dot(lime_att[...,:3], rgb_weights)
    
    
    # atts={'deconv':abs(grayscale_att_deconv),'saliency':abs(grayscale_att_sal),'gradients':abs(grayscale_att_ig),
    #       'shap':abs(grayscale_att_shap),'backprop':abs(grayscale_att_backprp),'ix':abs(grayscale_att_ix),
    #       'lrp':abs(grayscale_att_lrp), 'lime': abs(grayscale_att_lime), 'lrp-ab': abs(lrp_ab)}

    atts={'deconv':abs(grayscale_att_deconv),'ig':abs(grayscale_att_ig), 'shap':abs(grayscale_att_shap),
          'lrp':abs(grayscale_att_lrp), 'lrp-ab': abs(lrp_ab)}


    return atts,Y_probs



split = 0
model_ind = 0

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


model_conf=load_model(f'./models/cnn_confounder_{split}_{model_ind}.pt').to(DEVICE).eval()
model_sup=load_model(f'./models/cnn_suppressor_{split}_{model_ind}.pt').to(DEVICE).eval()
model_no=load_model(f'./models/cnn_no_watermark_{split}_{model_ind}.pt').to(DEVICE).eval()


def get_atts(data_watermark, data_no_watermark, ind):
    #target (same for both images)
    target=watermark_dataset[ind][1]

    return {'watermark_conf':plot_atts(data_watermark,model_conf,target),
        'watermark_sup':plot_atts(data_watermark,model_sup,target),
        'watermark_no':plot_atts(data_watermark,model_no,target),
        'no_watermark_conf':plot_atts(data_no_watermark,model_conf,target),
        'no_watermark_sup':plot_atts(data_no_watermark,model_sup,target),
        'no_watermark_no':plot_atts(data_no_watermark,model_no,target)}


def rgb2gray(rgb):
    rgb = rgb.transpose(1,2,0)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gray2rgb(gray):

    rgb = np.stack((gray,)*3, axis=-1)

    return rgb

def plots(n,atts, watermark_image, no_watermark_image):
    #plots 
    fig, axs = plt.subplots(6, 9,figsize=(28, 21))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    font_size=25
    plt.rcParams['font.size'] = font_size
    plt.rc('axes', titlesize=font_size) #title
    
    
    original_imgs=[watermark_image,no_watermark_image, abs(rgb2gray(watermark_image) - rgb2gray(no_watermark_image)),
                   watermark_image,no_watermark_image, abs(rgb2gray(watermark_image) - rgb2gray(no_watermark_image)),
                   watermark_image,no_watermark_image,  abs(rgb2gray(watermark_image) - rgb2gray(no_watermark_image))]
    
    images=[atts['watermark_conf'][0],atts['no_watermark_conf'][0], atts['diff_conf'],
            atts['watermark_sup'][0],atts['no_watermark_sup'][0], atts['diff_conf'],
            atts['watermark_no'][0],atts['no_watermark_no'][0], atts['diff_conf'] ]

    
    labels=['Original', 'Deconv','Int. Grad.','Grad SHAP',r'LRP-$\epsilon$', r'LRP-$\alpha \beta$']
    cmap = 'magma'
    
    for i in range(9): #images watermark/no watermark
        if i == 2 or i == 5 or i == 8:
            axs[0,i].imshow(original_imgs[i], cmap=cmap)
        else:
            axs[0,i].imshow(original_imgs[i].transpose(1,2,0))
        axs[1,i].imshow(images[i]['deconv'],cmap=cmap)
        axs[2,i].imshow(images[i]['ig'],cmap=cmap)
        axs[3,i].imshow(images[i]['shap'],cmap=cmap)
        axs[4,i].imshow(images[i]['lrp'],cmap=cmap)
        axs[5,i].imshow(images[i]['lrp-ab'],cmap=cmap)
        for j in range(6): #original image + XAI methods
            axs[j,i].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

            if i==0:
                axs[j,0].set_ylabel(labels[j]) 
    
    #titles model type
    plt.figtext(0.25,0.9,"Model trained with \n Confounder dataset", va="center", ha="center", size=font_size)
    plt.figtext(0.53,0.9,"Model trained with \n Suppressor dataset", va="center", ha="center", size=font_size)
    plt.figtext(0.78,0.9,"Model trained with \n No Watermark dataset", va="center", ha="center", size=font_size)
    
    plt.savefig(f'./figures/qualitative_{n}.png', bbox_inches='tight')
    plt.savefig(f'./figures/qualitative_{n}_hires.png', bbox_inches='tight', dpi=300)
    
    
for ind in np.random.choice(len(watermark_dataset), size=10, replace=False):
    watermark_image=watermark_dataset[ind][0]    
    data_watermark=torch.tensor(watermark_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

    no_watermark_image=no_watermark_dataset[ind][0]
    data_no_watermark=torch.tensor(no_watermark_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

    atts=get_atts(data_watermark, data_no_watermark, ind)
    plots(ind,atts,DEVICE,0, data_watermark, data_no_watermark)