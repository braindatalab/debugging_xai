from pycocotools.coco import COCO
import numpy as np

import cv2

import scipy.stats


from torchvision import datasets

import time

from PIL import Image

import pickle as pkl

import time

from sklearn.model_selection import StratifiedShuffleSplit

import gc



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

labels = np.zeros((n,))
masks = np.zeros((n, im_size,im_size))



mu = 0.5
sigma = 0.2
N = im_size*im_size

lower = 0.2
upper = 0.8

def sort_ref(ref):
    ref_norm = ref.copy()
    return ref_norm[np.argsort(ref)]

ref_01 = sort_ref(scipy.stats.truncnorm.rvs((0.0-mu)/sigma,(1.0-mu)/sigma,loc=mu,scale=sigma,size=N))    
ref_19 = sort_ref(scipy.stats.truncnorm.rvs((0.1-mu)/sigma,(0.9-mu)/sigma,loc=mu,scale=sigma,size=N))
ref_28 = sort_ref(scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)) # Control
ref_37 = sort_ref(scipy.stats.truncnorm.rvs((0.3-mu)/sigma,(0.7-mu)/sigma,loc=mu,scale=sigma,size=N))
ref_46 = sort_ref(scipy.stats.truncnorm.rvs((0.4-mu)/sigma,(0.6-mu)/sigma,loc=mu,scale=sigma,size=N)) 


datasets = {
    'norm': {
        'ref': ref_28.copy(),
        # 'data': np.zeros((n, im_size,im_size,3))
    },
    'conf_1': {
        'ref_0': ref_19.copy(),
        'ref_1': ref_37.copy(),
        # 'data': np.zeros((n, im_size,im_size,3))
    },
    'conf_2': {
        'ref_0': ref_01.copy(),
        'ref_1': ref_46.copy(),
        # 'data': np.zeros((n, im_size,im_size,3))
    },
    'sup_1': {
        'ref_0': ref_19.copy(),
        'ref_1': ref_37.copy(),
        # 'data': np.zeros((n, im_size,im_size,3))
    },
    'sup_2': {
        'ref_0': ref_01.copy(),
        'ref_1': ref_46.copy(),
        # 'data': np.zeros((n, im_size,im_size,3))
    },
}

# Specify 

# Doing the shuffling once with the 'norm' data and then using the same indices for the other datasets in the weird loop below
data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)

data = np.zeros((n, im_size,im_size,3))
labels = np.zeros((n,))
labels[int(n_per_class/2):] = 1

train_indices, val_test_indices = list(data_splitter.split(X=data, y=labels))[0]

x_train = data[train_indices]
y_train = labels[train_indices]
x_val_test = data[val_test_indices]
y_val_test = labels[val_test_indices]

data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_indices, test_indices = list(data_splitter.split(X=x_val_test, y=y_val_test))[0]

x_val = x_val_test[val_indices]
y_val = y_val_test[val_indices]

x_test = x_val_test[test_indices]
y_test = y_val_test[test_indices]




for dataset, refs in datasets.items():
    j = 0
    data = np.zeros((n, im_size,im_size,3))
    for i, (key, value) in enumerate(animal_no_intersections.items()):
        if len(value) >= n_per_class:
            print(i, key, len(value))

            for id in value[:n_per_class]:
                img = coco_train.loadImgs(id)[0]
        
                I = cv2.resize(cv2.imread(f'./coco/train2017/{img["file_name"]}', cv2.IMREAD_COLOR), (im_size,im_size))

                out_norm = np.asarray(cv2.cvtColor(   I, cv2.COLOR_RGB2HLS ))  # a bitmap conversion 
                out_norm = out_norm / 255

                
                copied = out_norm.copy()
                lightness_flattened = copied[:,:,1].reshape((im_size*im_size,))
                if dataset == 'norm':
                    lightness_flattened[np.argsort(lightness_flattened)] = refs['ref']
                elif 'conf' in dataset:
                    lightness_flattened[np.argsort(lightness_flattened)] = refs[f'ref_{i}']
                else:
                    if j >= int(n_per_class/2):
                        lightness_flattened[np.argsort(lightness_flattened)] = refs[f'ref_1']
                    else:
                        lightness_flattened[np.argsort(lightness_flattened)] = refs[f'ref_0']
                copied[:,:,1] = lightness_flattened.reshape((im_size,im_size))
                data[j] = copied
                
                annotation_ids = coco_train.getAnnIds(imgIds=id, catIds=[], iscrowd=False)
                annotations = coco_train.loadAnns(annotation_ids)
                
                if dataset == 'norm': 
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
                    
                    labels[j] = i
                    masks[j] = np.asarray(mask_im)[:,:,0]
                
                j+=1
    x_train = data[train_indices]
    y_train = labels[train_indices]
    masks_train = masks[train_indices]

    x_val_test = data[val_test_indices]
    y_val_test = labels[val_test_indices]
    masks_val_test = masks[val_test_indices]

    x_val = x_val_test[val_indices]
    y_val = y_val_test[val_indices]
    masks_val = masks_val_test[val_indices]

    x_test = x_val_test[test_indices]
    y_test = y_val_test[test_indices]
    masks_test = masks_val_test[test_indices]

    split_dataset = [(x_train, y_train, masks_train), (x_val, y_val, masks_train), (x_test, y_test, masks_train)]

    with open(f'coco_data_{dataset}.pkl', 'wb') as f:
        pkl.dump(split_dataset, f)

    del data, x_train, y_train, masks_train, x_val_test, y_val_test, masks_val_test, x_val, y_val, masks_val, x_test, y_test, masks_test, split_dataset
    gc.collect()

# # Doing the shuffling once with the 'norm' data and then using the same indices for the other datasets in the weird loop below
# data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)

# data = datasets['norm']['data']

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

# for dataset, value in datasets.items():
#     if dataset != 'norm':
#         data = value['data']

#         x_train = data[train_indices]
#         y_train = labels[train_indices]

#         x_val = x_val_test[val_indices]
#         y_val = y_val_test[val_indices]

#         x_test = x_val_test[test_indices]
#         y_test = y_val_test[test_indices]

#         split_dataset = [(x_train, y_train, masks[train_indices]), (x_val, y_val, masks[val_indices]), (x_test, y_test, masks[test_indices])]

#         with open(f'coco_data_{dataset}.pkl', 'wb') as f:
#             pkl.dump(split_dataset, f)

print('time to pre-process and save coco datasets:', time.time() - t0)

