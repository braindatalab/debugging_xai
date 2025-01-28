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

# DEVICE = 'cpu'

from captum.attr import IntegratedGradients, GradientShap, Deconvolution, LRP, Lime

from zennit.composites import EpsilonAlpha2Beta1

split = sys.argv[1]
model_ind = sys.argv[2]
print(f'SPLIT: {split} MODEL IND: {model_ind}')

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


def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

variable_string = f'_{sys.argv[3]}' if sys.argv[3] == 'variable' else ''

# variable position WM data has an extra return arg (stored wm array)
try:
    with open(f'./artifacts/split_{split}_no_watermark{variable_string}_test.pkl', 'rb') as f:
        no_watermark_dataset, labels_test_no, _ = pickle.load(f)
        no_watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_no.flatten()[i]] for i,x in enumerate(no_watermark_dataset)]

    with open(f'./artifacts/split_{split}_all_watermark{variable_string}_test.pkl', 'rb') as f:
        watermark_dataset, labels_test, _ = pickle.load(f)
        watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test.flatten()[i]] for i,x in enumerate(watermark_dataset)]
except:
    with open(f'./artifacts/split_{split}_no_watermark{variable_string}_test.pkl', 'rb') as f:
        no_watermark_dataset, labels_test_no, _, _ = pickle.load(f)
        no_watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_no.flatten()[i]] for i,x in enumerate(no_watermark_dataset)]

    with open(f'./artifacts/split_{split}_all_watermark{variable_string}_test.pkl', 'rb') as f:
        watermark_dataset, labels_test, _, masks_wm = pickle.load(f)
        watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test.flatten()[i]] for i,x in enumerate(watermark_dataset)]


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
    energy=(watermark_energy/watermark_size)/(image_energy/image_size)
    
    # Gives energy(watermark) = 1.0
    # energy = (np.sum(att*bin_water)/att.shape[0]*att.shape[1]) / (np.sum(att)/att.shape[0]*att.shape[1])
    
    return energy

def load_trained(path):
    model = Net()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


energy_water_conf={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
energy_water_sup={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
energy_water_no={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}

energy_no_water_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': [],}
energy_no_water_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
energy_no_water_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}


explanations_water_conf={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
explanations_water_sup={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}
explanations_water_no={'deconv':[],'int_grads':[],'shap':[],'lrp':[], 'lrp_ab': []}

explanations_no_water_conf={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': [],}
explanations_no_water_sup={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}
explanations_no_water_no={'deconv':[], 'int_grads':[], 'shap':[], 'lrp':[], 'lrp_ab': []}


model_conf=load_trained(f'./models/cnn_confounder{variable_string}_{split}_{model_ind}.pt').eval().to(DEVICE)
model_sup=load_trained(f'./models/cnn_suppressor{variable_string}_{split}_{model_ind}.pt').eval().to(DEVICE)
model_no=load_trained(f'./models/cnn_no_watermark{variable_string}_{split}_{model_ind}.pt').eval().to(DEVICE)

folder=os.getcwd()+'/images'
print(folder)
t0=time.time()

N_test = 1800

wm_avg = np.zeros((N_test,3,128,128))
no_wm_avg = np.zeros((N_test,3,128,128))

for i in range(N_test):
    wm_avg[i] = watermark_dataset[i][0]
    no_wm_avg[i] = no_watermark_dataset[i][0]
    
wm_avg = np.mean(wm_avg, axis=0)
no_wm_avg = np.mean(no_wm_avg, axis=0)

rgb_weights = [0.2989, 0.5870, 0.1140]


res = {
    'laplace': [[], []],
    'sobel': [[], []],
    'x': [[], []]
}


for i in range(len(watermark_dataset)):
    w_image = watermark_dataset[i]
    w_target=w_image[1]
    w_image=w_image[0]
    
    w_example=torch.tensor(w_image).unsqueeze(0).to(DEVICE,dtype=torch.float)

    w_target_conf = torch.max(model_conf(w_example), 1)[1].to(int)
    w_target_sup = torch.max(model_sup(w_example), 1)[1].to(int)
    w_target_no = torch.max(model_no(w_example), 1)[1].to(int)

    nw_image = no_watermark_dataset[i]
    nw_target=nw_image[1]
    nw_image=nw_image[0]
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
    
    # # explanations for ground truth class (n)w_target
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

        
        explanations_water_conf[method].append(a_conf_w[method])
        explanations_water_sup[method].append(a_sup_w[method])
        explanations_water_no[method].append(a_no_w[method])

        explanations_no_water_conf[method].append(a_conf_nw[method])
        explanations_no_water_sup[method].append(a_sup_nw[method])
        explanations_no_water_no[method].append(a_no_nw[method])

        # energy_water_conf_gt[method].append(energy(a_conf_w_gt[method]))
        # energy_water_sup_gt[method].append(energy(a_sup_w_gt[method]))
        # energy_water_no_gt[method].append(energy(a_no_w_gt[method]))

        # energy_no_water_conf_gt[method].append(energy(a_conf_nw_gt[method]))
        # energy_no_water_sup_gt[method].append(energy(a_sup_nw_gt[method]))
        # energy_no_water_no_gt[method].append(energy(a_no_nw_gt[method]))

    x_wm = energy(np.dot(w_image.copy().transpose(1,2,0)[...,:3], rgb_weights))
    x_no = energy(np.dot(nw_image.copy().transpose(1,2,0)[...,:3], rgb_weights))

    sample = w_image.copy().transpose(1,2,0) - wm_avg.transpose(1,2,0)
    img_r = sample[:,:,0]
    img_g = sample[:,:,1]
    img_b = sample[:,:,2]
    
    lapl_wm = energy(np.abs(laplace(img_r)) + np.abs(laplace(img_g)) + np.abs(laplace(img_b)))
    sob_wm = energy(np.abs(sobel(img_r)) + np.abs(sobel(img_g)) + np.abs(sobel(img_b)))

    # nw_image.append(energy(np.dot(nw_image.copy().transpose(1,2,0)[...,:3], rgb_weights)))
    
    sample = nw_image.copy().transpose(1,2,0) - no_wm_avg.transpose(1,2,0)
    img_r = sample[:,:,0]
    img_g = sample[:,:,1]
    img_b = sample[:,:,2]
    
    lapl_no = energy(np.abs(laplace(img_r)) + np.abs(laplace(img_g)) + np.abs(laplace(img_b)))
    sob_no = energy(np.abs(sobel(img_r)) + np.abs(sobel(img_g)) + np.abs(sobel(img_b)))
    

    res['laplace'][0].append(lapl_wm)
    res['laplace'][1].append(lapl_no)
    res['sobel'][0].append(sob_wm)
    res['sobel'][1].append(sob_no)
    res['x'][0].append(x_wm)
    res['x'][1].append(x_no)

    if (i%100)==0:
        print(i, 'out of', len(watermark_dataset))
        print(time.time()-t0)
        # t0=time.time()
    

for baseline, results in res.items():
        energy_water_conf[baseline] = results[0]
        energy_no_water_conf[baseline] = results[1]

        energy_water_sup[baseline] = results[0]
        energy_no_water_sup[baseline] = results[1]

        energy_no_water_conf[baseline] = results[0]
        energy_no_water_no[baseline] = results[1]


with open(f'./energies/energy_water_conf_pred{variable_string}_{split}_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_conf, f)    
with open(f'./energies/energy_water_sup_pred{variable_string}_{split}_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_sup, f)    
with open(f'./energies/energy_water_no_pred{variable_string}_{split}_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_water_no, f)    
    
with open(f'./energies/energy_no_water_conf_pred{variable_string}_{split}_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_conf, f)    
with open(f'./energies/energy_no_water_sup_pred{variable_string}_{split}_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_sup, f)    
with open(f'./energies/energy_no_water_no_pred{variable_string}_{split}_{model_ind}.pickle', 'wb') as f:
    pickle.dump(energy_no_water_no, f)

from sklearn import cluster, decomposition


config_strings = ['water_conf', 'no_conf', 'water_sup', 'no_sup', 'water_no', 'no_no']

for i, explanations in enumerate([explanations_water_conf, explanations_no_water_conf, explanations_water_sup, explanations_no_water_sup, explanations_water_no, explanations_no_water_no]):
    
    for method in list(explanations.keys()):
        print(f'{method} explanations PCA for {config_strings[i]} {sys.argv[3]} model split {split}')

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

        with open(f'./pcs/pc_{config_strings[i]}{variable_string}_{method}_{split}_{model_ind}.pickle', 'wb') as f:
            pickle.dump(components, f)


