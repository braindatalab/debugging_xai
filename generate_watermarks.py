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
    

def rescale_values(image,max_val,min_val):
    '''
    image - numpy array
    max_val/min_val - float
    '''
    return (image-image.min())/(image.max()-image.min())*(max_val-min_val)+min_val

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
intensity=0.8

def add_watermark(background_image_path,watermark_path, intensity_watermark,image_size,white_bool):
    #watermark should be jpg with white background
    #lower intensity_watermark leads to a more transparent watermark in the final image (less contrast)
    # white_bool is a bool that indicated if the watermark to add is white or black
    
    background_image = Image.open(background_image_path)
    background_image = background_image.resize(image_size)
    watermark = Image.open(watermark_path)

    #resize watermark to cover the entire background width without getting deformed
    w=int(background_image.size[0])
    h=int(watermark.size[1]*background_image.size[0]/watermark.size[0])
    watermark = watermark.resize((w,h))

    background_image=np.array(background_image)
    watermark=np.array(watermark)

    b=rescale_values(background_image,1,0)
    rgb=rescale_values(watermark,1,0)

    #turning watermark into grayscale
    r, g, blue = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 1-(0.2989 * r + 0.5870 * g + 0.1140 * blue)

    #padding watermark to be an image the same size as the background image
    white=np.ones((b.shape[0],b.shape[1]))
    white[0:gray.shape[0],0:gray.shape[1]]=gray

    # grayscale watermark with 3 channels
    gr=np.repeat(white[..., np.newaxis], 3, axis=2)

    if white_bool:
        im=1-b
        gr_p=rescale_values(gr,1,1-intensity_watermark)
        i_2=im*gr_p
        output_image=1-(i_2)
    else:
        im=rescale_values(b,1,0)
        gr_p=rescale_values(gr,1,1-intensity_watermark)
        output_image=im*gr_p
     
    #output_image=0
    return output_image

def save_images(wm_prev_cat, wm_prev_dog,cat_files,dog_files,watermark_path,image_size,intensity_watermark,output_path,white_bool):
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
    labels = np.zeros((len(cat_files) + len(dog_files), 1))
    labels[len(cat_files):] = 1
    watermark_inds = [with_watermark_cat, with_watermark_dog]

    for i, image in enumerate(cat_files):
        if image in with_watermark_cat:
            out_im=add_watermark(image,watermark_path,intensity_watermark,image_size,white_bool)
            n_water+=1
        else:
            out_im=Image.open(image)
            out_im = out_im.resize(image_size)
            out_im=np.array(out_im)
            n_no_water+=1
            
        #n=image.rfind('\\')
        # plt.imsave(output_path+image[n+1:],out_im)
        data[i] = out_im

    n_water_dog = 0
    n_no_water_dog = 0

    for i, image in enumerate(dog_files):
        if image in with_watermark_dog:
            out_im=add_watermark(image,watermark_path,intensity_watermark,image_size,white_bool)
            n_water_dog+=1
        else:
            out_im=Image.open(image)
            out_im = out_im.resize(image_size)
            out_im=np.array(out_im)
            n_no_water_dog+=1
            
        #n=image.rfind('\\')
        # plt.imsave(output_path+image[n+1:],out_im)
        data[len(cat_files) + i] = out_im

    
    print('number of images with watermark:',n_water, n_water_dog)
    print('number of images without watermark:',n_no_water, n_no_water_dog)

    with open(f'{output_path}.pkl', 'wb') as f:
            pickle.dump([data, labels, watermark_inds], f)
        
    return data


SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]
N = 6000 # per class

for i in [int(sys.argv[1])]:
    print("Generating data for split", i)
    SEED=SEEDS[i] 
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)

    
    inds = list(range(N))
    print("inds ", len(inds)) 
    train_inds = random.sample(inds, k=int(N*0.7))
    inds = list(np.delete(inds, np.where(train_inds==inds)))
    print("inds val", len(inds))
    val_inds = random.sample(inds, k=int(N*0.15))
    inds = list(np.delete(inds, np.where(val_inds==inds)))
    print("inds test", len(inds))
    test_inds = random.sample(inds, k=int(N*0.15))

    print(len(train_inds))
    print(len(val_inds))
    print(len(test_inds))
    #sys.exit()

    cat_names_train = list(np.array(cat_paths)[train_inds])
    cat_names_val = list(np.array(cat_paths)[val_inds])
    cat_names_test = list(np.array(cat_paths)[test_inds])
    
    dog_names_train = list(np.array(dog_paths)[train_inds])
    dog_names_val = list(np.array(dog_paths)[val_inds])
    dog_names_test = list(np.array(dog_paths)[test_inds])

    output_path=f'./artifacts/split_{i}_suppressor_'
    save_images(0.5, 0.5, cat_names_train, dog_names_train, watermark_path_jpeg,image_size,intensity,output_path+'train',1)
    save_images(0.5, 0.5, cat_names_val, dog_names_val, watermark_path_jpeg,image_size,intensity,output_path+'val',1)
    save_images(0.5, 0.5, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+'test',1)


    output_path=f'./artifacts/split_{i}_confounder_'
    
    save_images(0.2, 0.8, cat_names_train, dog_names_train, watermark_path_jpeg,image_size,intensity,output_path+'train',1)
    save_images(0.2, 0.8, cat_names_val, dog_names_val, watermark_path_jpeg,image_size,intensity,output_path+'val',1)
    save_images(0.2, 0.8, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+'test',1)
    print()
    
    output_path=f'./artifacts/split_{i}_no_watermark_'
    
    save_images(0, 0, cat_names_train,dog_names_train, watermark_path_jpeg,image_size,intensity,output_path+'train',1)
    save_images(0, 0, cat_names_val, dog_names_val, watermark_path_jpeg,image_size,intensity,output_path+'val',1)
    save_images(0, 0, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+'test',1)
    print()

    output_path=f'./artifacts/split_{i}_all_watermark_'
    save_images(1, 1, cat_names_test, dog_names_test, watermark_path_jpeg,image_size,intensity,output_path+'test',1)
    print()
