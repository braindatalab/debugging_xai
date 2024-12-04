import os
import time
import pickle as pkl
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import cv2

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

split = sys.argv[1]

with open(f'./artifacts/split_{split}_coco_data_norm_test.pkl', 'rb') as f:
    [x_test, y_test, _, _] = pkl.load(f)

norm_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=False, num_workers=4)

with open(f'./artifacts/split_{split}_coco_data_conf_test.pkl', 'rb') as f:
    [x_test, y_test, _, _] = pkl.load(f)


conf_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=False, num_workers=4)

with open(f'./artifacts/split_{split}_coco_data_sup_test.pkl', 'rb') as f:
    [x_test, y_test, _, _] = pkl.load(f)

sup_loader = DataLoader(TensorDataset(torch.tensor(x_test.transpose(0,3,1,2)), torch.tensor(y_test)), batch_size=batch_size, shuffle=False, num_workers=4)

def channel_sum(attr, channel):
    return np.sum(np.abs(attr[channel]))/np.sum(np.abs(attr))

folder=os.getcwd()+'/images'
print(folder)
t0=time.time()

energies = []
energies_rgb = []

for i, test_loader in enumerate([conf_loader, sup_loader, norm_loader]):
    datasets_energies = [[], [], []]
    datasets_energies_rgb = [[], [], []]
    for x_batch, test_labels in test_loader:      
        for j in range(x_batch.shape[0]):
            rgb = np.transpose(cv2.cvtColor( (np.transpose(x_batch[j].detach().numpy(), (1,2,0))*255).astype(np.uint8)  , cv2.COLOR_HLS2RGB ), (2,0,1))
            for channel_ind in range(3):
                datasets_energies[channel_ind].append(channel_sum(x_batch[j].detach().numpy(), channel_ind))
                datasets_energies_rgb[channel_ind].append(channel_sum(rgb, channel_ind))

    energies.append(datasets_energies)
    energies_rgb.append(datasets_energies_rgb)

with open(f'energies_coco_x_{split}.pickle', 'wb') as f:
    pkl.dump(energies, f)

with open(f'energies_coco_x_rgb_{split}.pickle', 'wb') as f:
    pkl.dump(energies_rgb, f)