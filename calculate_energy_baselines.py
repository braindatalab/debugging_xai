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

# DEVICE = 'cpu'

from captum.attr import IntegratedGradients, GradientShap, Deconvolution, LRP

from zennit.composites import EpsilonAlpha2Beta1

split = sys.argv[1]
# model_ind = sys.argv[2]
print(f'SPLIT: {split}')

def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val


with open(f'./artifacts/split_{split}_no_watermark_test.pkl', 'rb') as f:
    no_watermark_dataset, labels_test_no, _ = pickle.load(f)
    no_watermark_dataset = [[rescale_values(x,1,0).transpose(2,0,1),labels_test_no.flatten()[i]] for i,x in enumerate(no_watermark_dataset)]

with open(f'./artifacts/split_{split}_all_watermark_test.pkl', 'rb') as f:
    watermark_dataset, labels_test, _ = pickle.load(f)
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

from scipy.ndimage import sobel, laplace

rgb_weights = [0.2989, 0.5870, 0.1140]

lapl_wm = []
sob_wm = []
x_wm = []

for x, label in watermark_dataset:
    
    x_wm.append(energy(np.dot(x.copy().transpose(1,2,0)[...,:3], rgb_weights)))
    
    sample = x.copy().transpose(1,2,0) - wm_avg.transpose(1,2,0)
    img_r = sample[:,:,0]
    img_g = sample[:,:,1]
    img_b = sample[:,:,2]
    
    lapl = np.abs(laplace(img_r)) + np.abs(laplace(img_g)) + np.abs(laplace(img_b))
    sob = np.abs(sobel(img_r)) + np.abs(sobel(img_g)) + np.abs(sobel(img_b))
    
    lapl_wm.append(energy(lapl))
    sob_wm.append(energy(sob))
    
lapl_no = []
sob_no = []
x_no = []

for x, label in no_watermark_dataset:
    
    x_no.append(energy(np.dot(x.copy().transpose(1,2,0)[...,:3], rgb_weights)))
    
    sample = x.copy().transpose(1,2,0) - no_wm_avg.transpose(1,2,0)
    img_r = sample[:,:,0]
    img_g = sample[:,:,1]
    img_b = sample[:,:,2]
    
    lapl = np.abs(laplace(img_r)) + np.abs(laplace(img_g)) + np.abs(laplace(img_b))
    sob = np.abs(sobel(img_r)) + np.abs(sobel(img_g)) + np.abs(sobel(img_b))
    
    lapl_no.append(energy(lapl))
    sob_no.append(energy(sob))
       
with open(f'energy_baselines_{split}.pickle', 'wb') as f:
    pickle.dump([[lapl_no, lapl_wm], [sob_no, sob_wm], [x_no, x_wm]], f) 
    