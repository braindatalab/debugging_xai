import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
# plt.style.use('seaborn-colorblind')

# import pandas as pd

import pickle as pkl

from math import floor

import seaborn as sns

from pathlib import Path

from glob import glob
import scipy
import cv2


with open(f'./artifacts/coco_data_norm_test.pkl', 'rb') as f:
    [_, y_test_sample, _, _] = pkl.load(f)

N_samples = 20

test_0 = np.where(y_test_sample==0)[0]
test_1 = np.where(y_test_sample==1)[0]

test_sample_0 = np.random.choice(test_0, N_samples)
test_sample_1 = np.random.choice(test_1, N_samples)

for dataset in ['conf', 'sup', 'norm']:
    print(f'Loading data for coco dataset {dataset}...')

    with open(f'./artifacts/coco_data_{dataset}_test.pkl', 'rb') as f:
        [x_test, y_test, masks_test, _] = pkl.load(f)

    print(f'Plotting coco dataset {dataset}...')

    fig, axs = plt.subplots(4,N_samples, figsize=(30,15))


    for i in range(N_samples):
        # axs[0,i].imshow((x_test[test_sample_0[i]]*255).astype(np.uint8))
        # axs[0,i].set_title(f'Train 0: {test_sample_0[i]}')
        # axs[1,i].imshow((x_test[test_sample_1[i]]*255).astype(np.uint8))
        # axs[1,i].set_title(f'Train 1: {test_sample_1[i]}')        
        
        axs[0,i].imshow(cv2.cvtColor( (x_test[test_sample_0[i]]*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ) )
        axs[0,i].set_title(f'Train 0: {test_sample_0[i]}')

        a, b = 3,3
        # mean, var, skew, kurt = scipy.stats.beta(a, b, moments='mvsk')
        x = np.linspace(scipy.stats.beta.ppf(0.01, a, b),
                        scipy.stats.beta.ppf(0.99, a, b), 100)

        line1 = sns.histplot(x_test[test_sample_0[i]][:,:,1].flatten(), kde=True,stat="density", kde_kws=dict(cut=3), ax=axs[1,i])
        pdf = axs[1,i].plot(x, scipy.stats.beta.pdf(x, a, b), color='r', alpha=0.3)


        axs[2,i].imshow(cv2.cvtColor( (x_test[test_sample_1[i]]*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ))
        axs[2,i].set_title(f'Train 1: {test_sample_1[i]}')

        line1 = sns.histplot(x_test[test_sample_1[i]][:,:,1].flatten(), kde=True,stat="density", kde_kws=dict(cut=3), ax=axs[3,i])
        pdf = axs[3,i].plot(x, scipy.stats.beta.pdf(x, a, b), color='r', alpha=0.3)

    
    print(f'Plot done! Saving...')

    plt.savefig(f'data_coco_{dataset}.png', format='png', bbox_inches='tight')
