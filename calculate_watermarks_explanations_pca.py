import os
import sys
import numpy as np
import time
import pickle
from torch import manual_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import Image 

from scipy.ndimage import sobel, laplace

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

SEED=SEEDS[int(sys.argv[1])] 

manual_seed(SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

# if sys.argv[2] is not None:
#     DEVICE = torch.device("cuda",int(sys.argv[2]))
# else:
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = 'cpu'
print('testing testing 123')
print(DEVICE)

# DEVICE = 'cpu'

from captum.attr import IntegratedGradients, GradientShap, Deconvolution, LRP, Lime

from zennit.composites import EpsilonAlpha2Beta1

split = sys.argv[1]
model_name = sys.argv[2]
model_wm_placement = sys.argv[2] # fixed/variable; Whether the model is trained on fixed or variable watermark; gets tested on both fixed and variable test sets in this script
print(f'SPLIT: {split} TRAINED ON {model_name} data with {model_wm_placement} WM POSITION')

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

        grad = torch.eye(2, device=device)[[target]].to(device)
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

    target = int(target)
    
    ig_att = np.transpose(IntegratedGradients(model).attribute(data, target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # sal_att = np.transpose(Saliency(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    gradshap_att = np.transpose(GradientShap(model).attribute(data,target=target, baselines=torch.zeros(data.shape).to(DEVICE)).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # backprop_att = np.transpose(GuidedBackprop(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    # ix_att = np.transpose(InputXGradient(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    deconv_att = np.transpose(Deconvolution(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    lrp_att=np.transpose(LRP(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    lime_att=np.transpose(Lime(model).attribute(data,target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
    
    lrp_ab = lrp(data,model,target, DEVICE)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    grayscale_att_deconv = np.dot(deconv_att[...,:3], rgb_weights)
    # grayscale_att_ix = np.dot(ix_att[...,:3], rgb_weights)
    # grayscale_att_backprp = np.dot(backprop_att[...,:3], rgb_weights)
    grayscale_att_shap = np.dot(gradshap_att[...,:3], rgb_weights)
    # grayscale_att_sal = np.dot(sal_att[...,:3], rgb_weights)
    grayscale_att_ig = np.dot(ig_att[...,:3], rgb_weights)
    grayscale_att_lrp = np.dot(lrp_att[...,:3], rgb_weights)
    grayscale_att_lime = np.dot(lime_att[...,:3], rgb_weights)
    
    
    atts={
        'deconv':abs(grayscale_att_deconv),
        #   'saliency':abs(grayscale_att_sal),
          'int_grads':abs(grayscale_att_ig),
          'shap':abs(grayscale_att_shap),
        #   'backprop':abs(grayscale_att_backprp),
        #   'ix':abs(grayscale_att_ix),
          'lrp':abs(grayscale_att_lrp), 
          'lrp_ab':abs(lrp_ab),
          'lime': abs(grayscale_att_lime)
          }
    
    return atts,Y_probs

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model

# 
explanations_water={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []} # explanations for 
explanations_no_water={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': [],}

explanations_water_variable={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
explanations_no_water_variable={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': [],}

variable_str_model = ''
if model_wm_placement == 'variable':
     variable_str_model = '_variable'


folder=os.getcwd()+'/images'
print(folder)
t0=time.time()

rgb_weights = [0.2989, 0.5870, 0.1140]

for placement in ['fixed', 'variable']: # Each model trained on {conf, sup, no} with {fixed, variable} wm position gets tested on {fixed, variable} wm position data
    variable_str = ''
    if placement == 'variable':
        variable_str = '_variable'

    no_wm_path = f'./artifacts/split_{split}_no_watermark{variable_str}_test.pkl' 
    wm_path = f'./artifacts/split_{split}_all_watermark{variable_str}_test.pkl' 

    try:
        with open(no_wm_path, 'rb') as f:
                no_watermark_dataset, labels_test_no, _, _ = pickle.load(f)
                no_watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_no.flatten()[i]] for i,x in enumerate(no_watermark_dataset)]

        with open(wm_path, 'rb') as f:
                watermark_dataset, labels_test, _, _ = pickle.load(f)
                watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test.flatten()[i]] for i,x in enumerate(watermark_dataset)]
    except:
        with open(no_wm_path, 'rb') as f:
                no_watermark_dataset, labels_test_no, _ = pickle.load(f)
                no_watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_no.flatten()[i]] for i,x in enumerate(no_watermark_dataset)]

        with open(wm_path, 'rb') as f:
                watermark_dataset, labels_test, _ = pickle.load(f)
                watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test.flatten()[i]] for i,x in enumerate(watermark_dataset)]


    for model_ind in range(1):
        model=load_trained(f'./models/cnn_{model_name}{variable_str_model}_{split}_{model_ind}.pt').eval().to(DEVICE)

        for i in range(len(watermark_dataset)):
            w_image = watermark_dataset[i]
            w_target=w_image[1]
            w_image=w_image[0]
            w_example=torch.tensor(w_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

            w_target = torch.max(model(w_example), 1)[1].to(int)

            nw_image = no_watermark_dataset[i]
            nw_target=nw_image[1]
            nw_image=nw_image[0]
            nw_example=torch.tensor(nw_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

            nw_target = torch.max(model(nw_example), 1)[1].to(int)

            # explanations for predicted class (n)w_target_{MODEL}
            explanations_w, _ = plot_atts(w_example,model,w_target)
            explanations_nw, _ = plot_atts(nw_example,model,nw_target)

            
            for method in list(explanations_water.keys()):
                
                if placement == 'fixed':
                    explanations_water[method].append(explanations_w[method])
                    explanations_no_water[method].append(explanations_nw[method])
                else:
                    explanations_water_variable[method].append(explanations_w[method])
                    explanations_no_water_variable[method].append(explanations_nw[method])


import logging
import matplotlib.pyplot as plt

from sklearn import cluster, decomposition
from sklearn.preprocessing import MinMaxScaler

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def plot_gallery(title, images, image_shape=(128,128), n_col=2, n_row=5, out_file='', cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            # vec.reshape(image_shape).astype(np.int64),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.savefig(out_file)


water_strings = ['watermark', 'no watermark', 'watermark', 'no watermark']
position_strings = ['fixed', 'fixed', 'variable', 'variable']

rows, columns = 2, 5
n_plot = 10

for i, explanations in enumerate([explanations_water, explanations_no_water, explanations_water_variable, explanations_no_water_variable]):
    
    for method in list(explanations.keys()):
        print(f'{method} explanations PCA for {model_name} {model_wm_placement} model split {split} tested over {position_strings[i]} {water_strings[i]}')

        explanations_flattened = np.asarray(explanations[method]).reshape((np.asarray(explanations[method]).shape[0], -1))
        n_samples, n_features = explanations_flattened.shape
        print(n_samples, n_features)
        # Global centering (focus on one feature, centering all samples)
        explanations_centered = explanations_flattened - explanations_flattened.mean(axis=0)

        # Local centering (focus on one sample, centering all features)
        explanations_centered -= explanations_centered.mean(axis=1).reshape(n_samples, -1)

        # 0 < n_components < 1 => n_components% variance explained, aka 0.98 => 98% of variance must be explained by the number of components subsequently chosen
        pca_estimator = decomposition.PCA(
            n_components=0.98,  svd_solver="full"
        )
        pca_estimator.fit(explanations_centered)
        components = pca_estimator.components_
        print(components.shape)
        print(f'Number of PCA components: {pca_estimator.components_}')

        # eigenvectors = components.reshape((components.shape[0],128,128))

        plot_gallery(
            f"top {n_plot} PCA components for {method} explanations of {model_name} {model_wm_placement} model split {split} tested over {position_strings[i]} {water_strings[i]}", components[:n_plot], n_col=columns, n_row=rows, out_file=f'./figures/principal_components_{method}_{model_name}_{model_wm_placement}_{split}_{position_strings[i]}_{water_strings[i]}.png'
        )