import os
import sys
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image 

import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing import Process, Pool
from torch.nn.parallel import DistributedDataParallel as DDP
from captum._utils.models.linear_model import SkLearnLasso

SEED=1234
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
USE_CUDA = True
WORLD_SIZE = 4

from captum.attr import Lime

model_ind = sys.argv[1]
print(f'MODEL IND: {model_ind}')

import warnings
warnings.filterwarnings('ignore')

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

def get_lime_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return Lime(model).attribute(data, target=target, baselines=baselines)


methods_dict = {
    'lime': get_lime_attributions
}

#     return atts,Y_probs

rgb_weights = [0.2989, 0.5870, 0.1140]

def get_atts(data, model, target, methods=['deconv', 'int_grads', 'shap', 'lrp', 'lime']):
    atts = {}
    for method in methods:
        if 'deconv' in method or 'lrp' in method:
            att = np.transpose(methods_dict.get(method)(data=data,target=target,model=model).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
            # np.transpose(IntegratedGradients(model).attribute(data, target=target).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()
        else:
            att = np.transpose(methods_dict.get(method)(data=data,target=target,model=model, baselines=torch.zeros(data.shape).to(DEVICE)).squeeze().cpu().detach().numpy(), (1, 2, 0)).squeeze()

        atts[method] = abs(np.dot(att[...,:3], rgb_weights))
    
    return atts


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

def calculate_energy(dataset_name, dataset, model_name):
    energy_results={'lime':[]}
    model=load_trained(f'./cnn_{model_name}_{model_ind}.pt').to(DEVICE)
    folder=os.getcwd()+'/images'
    print(folder)
    t0=time.time()
    for i,w_image in enumerate(dataset):
        image=w_image[0]
        target=w_image[1]
        example=torch.tensor(image).unsqueeze(0).to(DEVICE,dtype=torch.float)
        explanations = get_atts(example, model, target)
        for method in list(explanations.keys()):
            energy_results[method].append(energy(explanations[method]))

        if (i%100)==0:
            print(i, 'out of', len(watermark_dataset))
            print(time.time()-t0)
            t0=time.time()
    
    return [model_name, dataset_name, energy_results] 


def run(rank, size, inp_batch, tar_batch, model_name, model_ind):
    # Initialize model
    model=load_trained(f'./cnn_{model_name}_{model_ind}.pt').to(DEVICE)
    
    # Move model and input to device with ID rank if USE_CUDA is True
    if USE_CUDA:
        inp_batch = inp_batch.cuda(rank)
        model = model.cuda(rank)
        # Uncomment line below to wrap with DistributedDataParallel
        model = DDP(model, device_ids=[rank])

    # Create IG object and compute attributions.
    # ig = Lime(model)
    attr = ig.attribute(inp_batch, target=0)
    lime =  Lime(model, interpretable_model=SkLearnLasso(alpha=0.0))
    attrs = []
    for i, inp in enumerate(inp_batch):
        example=torch.tensor(inp).unsqueeze(0).to(DEVICE,dtype=torch.float)
        target = tar_batch[i]
        attr = lime.attribute(example, target)
        attrs.append(attr)
        

    # Combine attributions from each device using distributed.gather
    # Rank 0 process gathers all attributions, each other process
    # sends its corresponding attribution.
    if rank == 0:
        output_list = [torch.zeros_like(torch.tensor(attrs)) for _ in range(size)]
        torch.distributed.gather(torch.tensor(attrs), gather_list=output_list)
        combined_attr = torch.cat(output_list)
        # Rank 0 prints the combined attribution tensor after gathering
        print(combined_attr)
    else:
        torch.distributed.gather(torch.tensor(attrs))

def init_process(rank, size, fn, inp_batch, tar_batch, model_name, model_ind, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, inp_batch, tar_batch, model_name, model_ind)
    dist.destroy_process_group()


size = WORLD_SIZE
processes = []

n_test = 12

watermark_chunked = torch.as_tensor(np.stack(np.array(watermark_dataset)[:n_test][:,0]), dtype=torch.float).chunk(WORLD_SIZE)
watermark_labels_chunked = torch.as_tensor(np.stack(np.array(watermark_dataset, dtype=object)[:n_test][:,1]), dtype=int).chunk(WORLD_SIZE)

model_name = 'conf'
# model_ind = 0

# pool = Pool(processes=WORLD_SIZE)


# batch_chunks = batch.chunk(size)
for rank in range(size):
    p = Process(target=init_process, args=(rank, size, run, watermark_chunked[rank], watermark_labels_chunked[rank], model_name, model_ind))
    p.start()
    processes.append(p)

for p in processes:
    p.join()



# watermark_data = torch.tensor(np.vstack(np.array(watermark_dataset)[:n_test][:,0]))
# watermark_labels = torch.tensor(np.vstack(np.array(watermark_dataset)[:n_test][:,1]))
        
# model_names = ['conf', 'sup', 'no']



# datasets = {'watermark': watermark_dataset, 'no_watermark': no_watermark_dataset}

# pool = Pool(processes=6)

# args_list = []
# for model_name in model_names:
#     for dataset_name, dataset in datasets.items():
#         args = (dataset_name, dataset, model_name)
#         args_list.append(args)

# results = pool.starmap_async(calculate_energy, args_list)

# energy_results = {}

# for result in results.get():
#     if result[0] not in energy_results.keys():
#         energy_results[result[0]] = {}
    
#     if result[1] not in energy_results[result[0]].keys():
#         energy_results[result[0]][result[1]] = {}
    
#     energy_results[result[0]][result[1]] = result[2]


# with open(f'energy_results_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_results, f) 



# with open(f'energy_water_conf_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_conf, f)    
# with open(f'energy_water_sup_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_sup, f)    
# with open(f'energy_water_no_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_water_no, f)    
    
# with open(f'energy_no_water_conf_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_conf, f)    
# with open(f'energy_no_water_sup_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_sup, f)    
# with open(f'energy_no_water_no_{model_ind}.pickle', 'wb') as f:
#     pickle.dump(energy_no_water_no, f)