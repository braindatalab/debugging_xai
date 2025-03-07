from pycocotools.coco import COCO
import numpy as np
import cv2
import sys
import scipy.stats
from skimage.exposure import match_histograms
import time
from PIL import Image
import pickle as pkl
from sklearn.preprocessing import minmax_scale 

# SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304]

SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]
SEED=SEEDS[int(sys.argv[1])]

np.random.seed(SEED)
# torch.manual_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

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

print("Loading COCO data...")

dataDir='coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco_train=COCO(annFile)

# display COCO categories and supercategories
cats = coco_train.loadCats(coco_train.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

id_dict = {}

animal_category_ids = coco_train.getCatIds(supNms=['animal'])
vehicle_category_ids = coco_train.getCatIds(supNms=['vehicle'])

id_dict['animal'] = []
id_dict['vehicle'] = []

for cat_id in animal_category_ids:
    id_dict['animal'].extend(coco_train.getImgIds(catIds=cat_id))

for cat_id in vehicle_category_ids:
    id_dict['vehicle'].extend(coco_train.getImgIds(catIds=cat_id))

animal_intersections = {}
for key, value in id_dict.items():
    animal_intersections[key] = []
    for key_2, value_2 in id_dict.items():
        if key != key_2:
            animal_intersections[key].extend(x for x in list(set(value).intersection(value_2)) if x not in animal_intersections[key])

animal_no_intersections = {}
sum = 0
for key, value in animal_intersections.items():
    animal_no_intersections[key] = [x for x in id_dict[key] if x not in value]
    print(key, len(id_dict[key]), len(value), len(animal_no_intersections[key]))
    sum += len(animal_no_intersections[key])

# n = 20000
# n_per_class = 10000

im_size = 224
N = im_size*im_size

n_samples = 30000
n_per_class = int(n_samples/2)

# labels = np.zeros((n_samples,))
# masks = np.zeros((n_samples, im_size,im_size))

beta_dark = scipy.stats.beta.rvs(2, 4, size=N)
beta_light = scipy.stats.beta.rvs(4, 2, size=N) 
beta_norm = scipy.stats.beta.rvs(3, 3, size=N) 


def generate_data(inds, beta_norm, beta_dark, beta_light, beta_conf, manip_sup, manip_conf, class_name='animal', im_size=224, manip_conf_str="", feature_range=(0,1)):
    data_norm = np.zeros((len(inds), im_size,im_size,3))
    data_conf = np.zeros((len(inds), im_size,im_size,3))
    data_sup = np.zeros((len(inds), im_size,im_size,3))
    data_dark = np.zeros((len(inds), im_size,im_size,3))
    data_light = np.zeros((len(inds), im_size,im_size,3))

    masks = np.zeros((len(inds), im_size,im_size))

    manips_norm = []
    manips_conf = []
    manips_sup = []

    

    # for ind in inds

    for i, ind in enumerate(inds):
        img = coco_train.loadImgs([ind])[0]

        im_cv = cv2.resize(cv2.imread(f'./coco/train2017/{img["file_name"]}', cv2.IMREAD_COLOR), (im_size,im_size))

        # OpenCV defaults to loading images as BGR not RGB
        out_norm = np.asarray(cv2.cvtColor(   im_cv, cv2.COLOR_BGR2HLS ))  # a bitmap conversion BGR -> HLS
        out_norm = out_norm / 255


        # Norm
        copied = out_norm.copy()
        lightness_flattened = copied[:,:,1].reshape((im_size*im_size,))
        norm_scaled = minmax_scale(match_histograms(lightness_flattened.copy(), beta_norm.flatten()), feature_range=feature_range)
        copied[:,:,1] = norm_scaled.reshape((im_size,im_size))
        data_norm[i] = copied
        manips_norm.append('norm')

        copied = out_norm.copy()
        lightness_flattened = copied[:,:,1].reshape((im_size*im_size,))
        dark_scaled = minmax_scale(match_histograms(lightness_flattened.copy(), beta_dark.flatten()), feature_range=feature_range)
        copied[:,:,1] = dark_scaled.reshape((im_size,im_size))
        data_dark[i] = copied

        copied = out_norm.copy()
        lightness_flattened = copied[:,:,1].reshape((im_size*im_size,))
        light_scaled = minmax_scale(match_histograms(lightness_flattened.copy(), beta_light.flatten()), feature_range=feature_range)
        copied[:,:,1] = light_scaled.reshape((im_size,im_size))
        data_light[i] = copied
        
        # Conf
        if ind in manip_conf:
            copied_conf = out_norm.copy()
            lightness_flattened_conf = copied_conf[:,:,1].reshape((im_size*im_size,))
            lightness_flattened_conf = minmax_scale(match_histograms(lightness_flattened_conf, beta_conf.flatten()), feature_range=feature_range)
            copied_conf[:,:,1] = lightness_flattened_conf.reshape((im_size,im_size))
            data_conf[i] = copied_conf 
            manips_conf.append(manip_conf_str)
        else:
            data_conf[i] = data_norm[i]
            manips_conf.append('norm')

        # Sup
        if ind in manip_sup:
            # print(copied)
            copied_sup = out_norm.copy()
            lightness_flattened_sup = copied_sup[:,:,1].reshape((im_size*im_size,))
            if ind in manip_sup[:int(len(manip_sup)/2)]:
                lightness_flattened_sup = minmax_scale(match_histograms(lightness_flattened_sup, beta_dark.flatten()), feature_range=feature_range)
                manips_sup.append('dark')
            else:
                lightness_flattened_sup = minmax_scale(match_histograms(lightness_flattened_sup, beta_light.flatten()), feature_range=feature_range)
                manips_sup.append('light')
            copied_sup[:,:,1] = lightness_flattened_sup.reshape((im_size,im_size))
            data_sup[i] = copied_sup
        else:
            data_sup[i] = data_norm[i]
            manips_sup.append('norm')

        annotation_ids = coco_train.getAnnIds(imgIds=[ind], catIds=[], iscrowd=False)
        annotations = coco_train.loadAnns(annotation_ids)

        mask = np.zeros(coco_train.annToMask(annotations[0]).shape)
        for annotation in annotations:
            if class_name == 'animal':
                if annotation['category_id'] in animal_category_ids:
                    masks_animal = coco_train.annToMask(annotation)
                    mask += masks_animal
            elif class_name == 'vehicle':
                if annotation['category_id'] in vehicle_category_ids:
                    masks_animal = coco_train.annToMask(annotation)
                    mask += masks_animal
        mask[mask!=0] = 1

        mask_im = Image.fromarray(np.uint8(mask)).convert('RGB').resize((im_size,im_size))
        
        # labels[j] = i
        masks[i] = np.asarray(mask_im)[:,:,0]
    
    return data_norm, data_conf, data_sup, data_dark, data_light, masks, manips_norm, manips_conf, manips_sup


SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]


for i in [int(sys.argv[1])]:
    print("Generating data for split", i)
    SEED=SEEDS[i] 
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)

    inds = list(range(n_per_class))
    print("inds ", len(inds)) 
    # print(inds)

    # train_inds = random.sample(inds, k=int(N*0.7))
    train_inds = np.random.choice(inds,size=int(n_per_class*0.7), replace=False)
    inds = np.setdiff1d(inds, train_inds)
    print("inds val", len(inds))

    val_inds = np.random.choice(inds,size=int(n_per_class*0.15), replace=False)
    inds = np.setdiff1d(inds, val_inds)
    print("inds test", len(inds))

    test_inds = np.random.choice(inds,size=int(n_per_class*0.15), replace=False)

    feature_range = (0,1)
    rescaled_string = ''
    if len(sys.argv > 2):
        if sys.argv[2] == 'rescaled':
            feature_range = (-1,1)
            rescaled_string = '_rescaled'

    # This is terrible naming, sorry. split here = train/val/test; sys.argv[1] is for the split in the sense of shuffling of data)
    for split, split_results in {'train': [train_inds, 0.7], 'val': [val_inds, 0.15], 'test': [test_inds, 0.15]}.items():

        n_samples_split = int(n_samples*split_results[1])
        n_per_class_split = int(n_per_class*split_results[1])

        data_norm = np.zeros((n_samples_split, im_size,im_size,3))
        data_dark = np.zeros((n_samples_split, im_size,im_size,3))
        data_light = np.zeros((n_samples_split, im_size,im_size,3))
        data_conf = np.zeros((n_samples_split, im_size,im_size,3))
        data_sup = np.zeros((n_samples_split, im_size,im_size,3))
        masks = np.zeros((n_samples_split, im_size,im_size))
        
        half = int(len(split_results[0])*0.5)

        manips_norm = []
        manips_conf = []
        manips_sup = []
        
        manip_inds = np.zeros((int(half*2),))

        labels = np.zeros((n_samples_split,))
        labels[n_per_class_split:] = 1

        for j, (key, value) in enumerate(animal_no_intersections.items()):
            print(f'Processing {split}; {key}...')
            value = np.array(value)
            manip_half=np.random.choice(value[split_results[0]],size=half, replace=False)
            # manip_conf = np.random.choice(value[split_results[0]],size=int(len(split_results[0])*0.5), replace=False)

            if j == 0:
                beta_conf = beta_dark
                manip_conf_str = 'dark'
            else:
                beta_conf = beta_light
                manip_conf_str = 'light'

            data_norm_half, data_conf_half, data_sup_half, data_dark_half, data_light_half, masks_half, manips_norm_half, manips_conf_half, manips_sup_half = generate_data(value[split_results[0]][:n_per_class_split], beta_norm, beta_dark, beta_light, beta_conf, manip_half, manip_half, key, manip_conf_str=manip_conf_str, feature_range=feature_range)
            data_norm[j*n_per_class_split:(j+1)*n_per_class_split] = data_norm_half
            data_dark[j*n_per_class_split:(j+1)*n_per_class_split] = data_dark_half
            data_light[j*n_per_class_split:(j+1)*n_per_class_split] = data_light_half
            data_conf[j*n_per_class_split:(j+1)*n_per_class_split] = data_conf_half
            data_sup[j*n_per_class_split:(j+1)*n_per_class_split] = data_sup_half
            masks[j*n_per_class_split:(j+1)*n_per_class_split] = masks_half
            manip_inds[j*half:(j+1)*half] = manip_half

            manips_norm.extend(manips_norm_half)
            manips_conf.extend(manips_conf_half)
            manips_sup.extend(manips_sup_half)
        
        split_dataset = [(data_norm, data_conf, data_sup), masks]
        
        with open(f'./artifacts/split_{i}_coco_data_norm_{split}{rescaled_string}.pkl', 'wb') as f:
            pkl.dump([data_norm, labels, masks, manip_inds, manips_norm], f)

        with open(f'./artifacts/split_{i}_coco_data_conf_{split}{rescaled_string}.pkl', 'wb') as f:
            pkl.dump([data_conf, labels, masks, manip_inds, manips_conf], f)

        with open(f'./artifacts/split_{i}_coco_data_sup_{split}{rescaled_string}.pkl', 'wb') as f:
            pkl.dump([data_sup, labels, masks, manip_inds, manips_sup], f)

        if split == 'test':
            with open(f'./artifacts/split_{i}_coco_data_dark_{split}{rescaled_string}.pkl', 'wb') as f:
                pkl.dump([data_dark, labels, masks, manip_inds, manips_norm], f)

            with open(f'./artifacts/split_{i}_coco_data_light_{split}{rescaled_string}.pkl', 'wb') as f:
                pkl.dump([data_light, labels, masks, manip_inds, manips_norm], f)
    








