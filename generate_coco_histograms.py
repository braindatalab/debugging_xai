# %matplotlib inline
# %matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import cv2


import torch
from torch import nn
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, AUROC
import time
import torch.nn.functional as F

from PIL import Image

import pickle as pkl

import time

from sklearn.model_selection import StratifiedShuffleSplit
SEED = 1234

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    # https://en.wikipedia.org/wiki/Histogram_equalization
    # https://stackoverflow.com/a/28520445
    
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


t0 = time.time()


dataDir='coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco_train=COCO(annFile)

# dataDir='coco'
# dataType='val2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# coco_val = COCO(annFile)

# display COCO categories and supercategories
cats = coco_train.loadCats(coco_train.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

animal_id_dict = {}

animal_category_ids = coco_train.getCatIds(supNms=['animal'])
vehicle_category_ids = coco_train.getCatIds(supNms=['vehicle'])

animal_id_dict['animal'] = []
animal_id_dict['vehicle'] = []

for cat_id in animal_category_ids:
    animal_id_dict['animal'].extend(coco_train.getImgIds(catIds=cat_id))
#    print(len(animal_id_dict[cat_id]))

for cat_id in vehicle_category_ids:
    animal_id_dict['vehicle'].extend(coco_train.getImgIds(catIds=cat_id))
#    print(len(animal_id_dict[cat_id]))

# animal_id_dict_val = {}
# print('\n')
# for cat_id in animal_category_ids:
#     animal_id_dict_val[cat_id] = coco_val.getImgIds(catIds=cat_id)
#     print(len(animal_id_dict_val[cat_id]))


animal_intersections = {}
for key, value in animal_id_dict.items():
    animal_intersections[key] = []
    for key_2, value_2 in animal_id_dict.items():
        if key != key_2:
            animal_intersections[key].extend(x for x in list(set(value).intersection(value_2)) if x not in animal_intersections[key])

# animal_intersections_val = {}
# for key, value in animal_id_dict_val.items():
#     animal_intersections_val[key] = []
#     for key_2, value_2 in animal_id_dict_val.items():
#         if key != key_2:
#             animal_intersections_val[key].extend(x for x in list(set(value).intersection(value_2)) if x not in animal_intersections_val[key])

animal_no_intersections = {}
sum = 0
for key, value in animal_intersections.items():
    animal_no_intersections[key] = [x for x in animal_id_dict[key] if x not in value]
    print(key, len(animal_id_dict[key]), len(value), len(animal_no_intersections[key]))
    sum += len(animal_no_intersections[key])

# animal_no_intersections_val = {}
# sum_val = 0
# for key, value in animal_intersections_val.items():
#     animal_no_intersections_val[key] = [x for x in animal_id_dict_val[key] if x not in value]
#     print(key, len(animal_id_dict_val[key]), len(value), len(animal_no_intersections_val[key]))
#     sum_val += len(animal_no_intersections_val[key])


n = 20000
n_per_class = 10000

im_size = 224

data = np.zeros((n, im_size,im_size,3))
data_not_norm = np.zeros((n, im_size,im_size,3))
labels = np.zeros((n,))
masks = np.zeros((n, im_size,im_size))

j = 0
for i, (key, value) in enumerate(animal_no_intersections.items()):
    if len(value) >= n_per_class:
        print(i, key, len(value))

        for id in value[:n_per_class]:
            img = coco_train.loadImgs(id)[0]
    
            I = cv2.resize(cv2.imread(f'./coco/train2017/{img["file_name"]}', cv2.IMREAD_COLOR), (im_size,im_size))

            out_norm = np.asarray(cv2.cvtColor(   I, cv2.COLOR_RGB2HLS ))  # a bitmap conversion 
            data_not_norm[j] = out_norm.copy()
            out_norm[:,:,1] = image_histogram_equalization(out_norm[:,:,1])[0]
            out_norm = out_norm / 255
            
            annotation_ids = coco_train.getAnnIds(imgIds=id, catIds=[], iscrowd=False)
            annotations = coco_train.loadAnns(annotation_ids)
            
            mask = np.zeros(coco_train.annToMask(annotations[0]).shape)
            for annotation in annotations:
                # print(annotation)
                if key == 'animal':
                    if annotation['category_id'] in animal_category_ids:
                        masks_animal = coco_train.annToMask(annotation)
                        # print(masks)
                        mask += masks_animal
                elif key == 'vehicle':
                    if annotation['category_id'] in vehicle_category_ids:
                        masks_animal = coco_train.annToMask(annotation)
                        mask += masks_animal
            mask[mask!=0] = 1

            mask_im = Image.fromarray(np.uint8(mask)).convert('RGB').resize((im_size,im_size))
            
            data[j] = out_norm
            labels[j] = i
            masks[j] = np.asarray(mask_im)[:,:,0]
            
            j+=1


fig,axs = plt.subplots(1,2, figsize=(12,6))

axs[0].hist(data_not_norm[:,:,:,1].flatten(), bins=256)  # arguments are passed to np.histogram
axs[1].set_title("lightness distribution histogram")

axs[1].hist(data[:,:,:,1].flatten(), bins=256)  # arguments are passed to np.histogram
axs[1].set_title("Normalised lightness distribution histogram")

plt.savefig('lightness_histograms.png', bbox_inches='tight')


# def create_conf_data(data, n=20000, n_per_class=10000):
#     conf_data = data.copy()

#     # animals = brightness * 0.8
#     # vehicles = brightness * 0.5
#     # should I use the same brightness for all images of the given class?
#     # or should I do a 50% normalised 50% not normalised split?

#     return conf_data

# def create_sup_data(data, n=20000, n_per_class=10000):
#     sup_data = data.copy()

#     # random brightness - should I pick ~5 fixed brightness levels and evenly distribute, or should I use a continuous distribution?

#     return sup_data

# data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
# train_indices, val_indices = list(data_splitter.split(X=data, y=labels))[0]


# x_train = data[train_indices]
# y_train = labels[train_indices]
# x_val_test = data[val_indices]
# y_val_test = labels[val_indices]

# data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
# val_indices, test_indices = list(data_splitter.split(X=x_val_test, y=y_val_test))[0]

# x_val = x_val_test[val_indices]
# y_val = y_val_test[val_indices]

# x_test = x_val_test[test_indices]
# y_test = y_val_test[test_indices]


# split_dataset_norm = [(x_train, y_train, masks[train_indices]), (x_val, y_val, masks[val_indices]), (x_test, y_test, masks[test_indices])]

# with open('coco_data_norm.pkl', 'wb') as f:
#     pkl.dump(split_dataset_norm, f)


# print('time to pre-process and save coco datasets:', time.time() - t0)

