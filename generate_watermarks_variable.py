import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle

from os import listdir
from os.path import isfile, join
from pathlib import Path

import sys
import random


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gray2rgb(gray):

    rgb = np.stack((gray,)*3, axis=-1)

    return rgb
    

def rescale_values(image,max_val=1,min_val=0):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

def preprocess_watermark(watermark_path, image_shape=(128,128)):
    watermark = Image.open(watermark_path)

    #resize watermark to cover the entire background width without getting deformed
    w=int(image_shape[0])
    h=int(watermark.size[1]*image_shape[0]/watermark.size[0])
    watermark = np.array(watermark.resize((w,h)))


    #turning watermark into grayscale
    rgb=rescale_values(watermark,1,0)
    r, g, blue = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 1-(0.2989 * r + 0.5870 * g + 0.1140 * blue)


    # trim excess rows and cols
    non_trimmed_row_indices =[i for i in range(gray.shape[0]) if not np.allclose(gray[i,:],1)]
    watermark_trimmed = gray[non_trimmed_row_indices,:]

    non_trimmed_col_indices =[i for i in range(watermark_trimmed.shape[1]) if not np.allclose(watermark_trimmed[:,i],1)]
    watermark_trimmed = watermark_trimmed[:,non_trimmed_col_indices]

    return watermark_trimmed

def get_watermark_position(watermark_shape, image_shape=(128,128)):
    rand_y = np.random.randint(0, high= (image_shape[0])-watermark_shape[0])
    rand_x = np.random.randint(0, high= (image_shape[1])-watermark_shape[1])

    return rand_y, rand_x

SEED=0
np.random.seed(SEED)
import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

from glob import glob
# random.sample(cat_names,k=6000)
cat_paths = glob('./images/cat/*')
dog_paths = glob('./images/dog/*')
print('total dogs and cats:', len(dog_paths), len(cat_paths))

folder=''
watermark_path_jpeg=folder+'watermark banner.jpg'
image_size=(128,128)
intensity=0.8 # opacity

def add_watermark_variable(background_image_path,watermark_path, intensity_watermark,image_size,white_bool=1, max_val=1, min_val=0):
    #watermark should be jpg with white background
    #lower intensity_watermark leads to a more transparent watermark in the final image (less contrast)
    # white_bool is a bool that indicated if the watermark to add is white or black
    
    background_image = Image.open(background_image_path)
    background_image = background_image.resize(image_size)
    background_image=np.array(background_image)
    b=rescale_values(background_image,max_val,min_val)

    wm_preprocessed = preprocess_watermark(watermark_path)
    wm_shape = wm_preprocessed.shape
    pos = get_watermark_position(wm_shape)

    #padding watermark to be an image the same size as the background image
    white=np.ones((b.shape[0],b.shape[1]))
    white[pos[0]:pos[0]+wm_shape[0], pos[1]:pos[1]+wm_shape[1]] = wm_preprocessed

    # grayscale watermark with 3 channels
    gr=np.repeat(white[..., np.newaxis], 3, axis=2)

    if white_bool:
        im=1-b
        gr_p=rescale_values(gr,1,1-intensity_watermark)
        i_2=im*gr_p
        output_image=1-(i_2)
    else:
        im=rescale_values(b,max_val,min_val)
        gr_p=rescale_values(gr,1,1-intensity_watermark)
        output_image=im*gr_p
     
    #output_image=0
    return output_image, 1.0*(white<1)

def save_images(wm_prev_cat, wm_prev_dog,cat_files,dog_files,watermark_path,image_size,intensity_watermark,output_path,white_bool=1, max_val=1, min_val=0):
    with_watermark_cat=np.random.choice(cat_files,size=int(len(cat_files)*wm_prev_cat), replace=False)
    with_watermark_dog=np.random.choice(dog_files,size=int(len(dog_files)*wm_prev_dog), replace=False)

    print(f'creating dataset {output_path}')
    print("prevs", wm_prev_cat, wm_prev_dog)
    print('num cat', len(cat_files))
    print('num dog', len(dog_files))
    print('wm cat:', int(len(cat_files)*wm_prev_cat), len(with_watermark_cat))
    print('wm dog:', int(len(dog_files)*wm_prev_dog), len(with_watermark_dog))

    # Path(output_path).mkdir(parents=True, exist_ok=True)
    n_water=0
    n_no_water=0

    data = np.zeros((len(cat_files) + len(dog_files), 128, 128, 3))
    masks = np.zeros((len(cat_files) + len(dog_files), 128, 128))
    labels = np.zeros((len(cat_files) + len(dog_files), 1))
    labels[len(cat_files):] = 1

    watermark_inds = []
    data_ind = 0 

    for i, image in enumerate(cat_files):
        if image in with_watermark_cat:
            out_im, mask=add_watermark_variable(image,watermark_path,intensity_watermark,image_size,white_bool, max_val, min_val)
            n_water+=1
            watermark_inds.append(data_ind)
        else:
            out_im = Image.open(image)
            out_im = out_im.resize(image_size)
            out_im = rescale_values(np.array(out_im), max_val, min_val)
            n_no_water+=1
            mask = np.zeros((128,128))
        data_ind += 1
            
            
        #n=image.rfind('\\')
        # plt.imsave(output_path+image[n+1:],out_im)
        data[i] = out_im
        masks[i] = mask

    n_water_dog = 0
    n_no_water_dog = 0

    for i, image in enumerate(dog_files):
        if image in with_watermark_dog:
            out_im, mask=add_watermark_variable(image,watermark_path,intensity_watermark,image_size,white_bool, max_val, min_val)
            n_water_dog+=1
            watermark_inds.append(data_ind)
        else:
            out_im = Image.open(image)
            out_im = out_im.resize(image_size)
            out_im = rescale_values(np.array(out_im), max_val, min_val)
            n_no_water_dog+=1
            mask = np.zeros((128,128))
        data_ind += 1
            
        #n=image.rfind('\\')
        # plt.imsave(output_path+image[n+1:],out_im)
        data[len(cat_files) + i] = out_im
        masks[len(cat_files) + i] = mask

    
    print('number of images with watermark:',n_water, n_water_dog)
    print('number of images without watermark:',n_no_water, n_no_water_dog)

    with open(f'{output_path}.pkl', 'wb') as f:
            pickle.dump([data, labels, watermark_inds, masks], f)
        
    return data, masks


SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]
N = 6000 # per class

rescaled_string = ''
rescaled_bool = False
if len(sys.argv) > 2:
    if sys.argv[2] == 'rescaled':
        rescaled_string = '_rescaled'
        rescaled_bool = True


for i in [int(sys.argv[1])]:
    print("Generating data for split", i)
    SEED=SEEDS[i] 
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)

    
    inds = list(range(N))
    print("inds ", len(inds)) 
    # print(inds)

    # train_inds = random.sample(inds, k=int(N*0.7))
    train_inds = np.random.choice(inds,size=int(N*0.7), replace=False)
    inds = np.setdiff1d(inds, train_inds)
    print("inds val", len(inds))

    val_inds = np.random.choice(inds,size=int(N*0.15), replace=False)
    inds = np.setdiff1d(inds, val_inds)
    print("inds test", len(inds))

    test_inds = np.random.choice(inds,size=int(N*0.15), replace=False)

    cat_names_train = list(np.array(cat_paths)[train_inds])
    cat_names_val = list(np.array(cat_paths)[val_inds])
    cat_names_test = list(np.array(cat_paths)[test_inds])
    
    dog_names_train = list(np.array(dog_paths)[train_inds])
    dog_names_val = list(np.array(dog_paths)[val_inds])
    dog_names_test = list(np.array(dog_paths)[test_inds])

    output_path=f'./artifacts/split_{i}_suppressor_variable_'

    save_images(0.5, 0.5, cat_names_train, dog_names_train, watermark_path_jpeg,image_size,intensity,output_path+f'train{rescaled_string}',1, rescaled_bool)
    save_images(0.5, 0.5, cat_names_val, dog_names_val, watermark_path_jpeg,image_size,intensity,output_path+f'val{rescaled_string}',1, rescaled_bool)
    save_images(0.5, 0.5, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+f'test{rescaled_string}',1, rescaled_bool)


    output_path=f'./artifacts/split_{i}_confounder_variable_'
    
    save_images(0.2, 0.8, cat_names_train, dog_names_train, watermark_path_jpeg,image_size,intensity,output_path+f'train{rescaled_string}',1, rescaled_bool)
    save_images(0.2, 0.8, cat_names_val, dog_names_val, watermark_path_jpeg,image_size,intensity,output_path+f'val{rescaled_string}',1, rescaled_bool)
    save_images(0.2, 0.8, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+f'test{rescaled_string}',1, rescaled_bool)
    print()
    
    output_path=f'./artifacts/split_{i}_no_watermark_variable_'
    
    save_images(0, 0, cat_names_train,dog_names_train, watermark_path_jpeg,image_size,intensity,output_path+f'train{rescaled_string}',1, rescaled_bool)
    save_images(0, 0, cat_names_val, dog_names_val, watermark_path_jpeg,image_size,intensity,output_path+f'val{rescaled_string}',1, rescaled_bool)
    save_images(0, 0, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+f'test{rescaled_string}',1, rescaled_bool)
    print()

    output_path=f'./artifacts/split_{i}_all_watermark_variable_'
    save_images(1, 1, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+f'test{rescaled_string}',1, rescaled_bool)
    print()
