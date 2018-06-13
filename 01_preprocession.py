###################################################
#
#   Script to ready input files
#
##################################################

import numpy as np
from tqdm import tqdm
from PIL import Image
import h5py
from keras.preprocessing.image import load_img
from skimage.transform import resize
import sys

import warnings
warnings.filterwarnings('ignore')

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def seg_to_cat(seg,cat):
    # Update seg image to cat image with category ID
    cat_seg = []

    for row in seg:
        for i in row:
            #print(type(i))
            cat_seg.append(int(cat[0][int(i)])-1)

    cat_seg = np.array(cat_seg)
    cat_seg = cat_seg.reshape(IMG_HEIGHT, IMG_WIDTH)
    
    return cat_seg


if __name__ == '__main__':

    print("\n1.Extract images from .mat")

    filepath = './input/fashon_parsing_data.mat'
    arrays = {}
    f = h5py.File(filepath)

    for k, v in f.items():
        arrays[k] = np.array(v)
    
    print("{}".format(arrays.keys()))
    print('Done!')

    # Set some parameters
    IMG_WIDTH = 400
    IMG_HEIGHT = 608 #608#592
    IMG_CHANNELS = 3
    IMG_SEGMENTATION = 425
    N_Cls=23

    # Get train and test IDs
    ids = list(arrays['#refs#'])

    # Train : Valid : Test = 0.72 : 0.08 : 0.2
    train_ratio=0.9 * 0.9 #0.01#
    valid_ratio=0.9 * 0.1 #0.01#
    test_ratio=0.1 #test

    train_data_count=int(len(ids) * train_ratio)
    valid_data_count=int(len(ids) * valid_ratio)
    test_data_count=int(len(ids) * test_ratio)
    #test_data_count=len(ids)-train_data_count-valid_data_count

    train_ids=ids[:train_data_count]
    valid_ids=ids[train_data_count:train_data_count+valid_data_count]
    #test_ids=ids[train_data_count+valid_data_count:]
    test_ids=ids[train_data_count+valid_data_count:train_data_count+valid_data_count+test_data_count] #test

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, N_Cls))
    Y_original = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, N_Cls))
    col_train = np.zeros((len(train_ids), 1, IMG_SEGMENTATION))
    cat_train = np.zeros((len(train_ids), 1, IMG_SEGMENTATION))

    X_valid = np.zeros((len(valid_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_valid = np.zeros((len(valid_ids), IMG_HEIGHT, IMG_WIDTH, N_Cls))
    col_valid = np.zeros((len(valid_ids), 1, IMG_SEGMENTATION))
    cat_valid = np.zeros((len(valid_ids), 1, IMG_SEGMENTATION))

    X_test = np.zeros((len(valid_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((len(valid_ids), IMG_HEIGHT, IMG_WIDTH, N_Cls))
    col_test = np.zeros((len(valid_ids), 1, IMG_SEGMENTATION))
    cat_test = np.zeros((len(valid_ids), 1, IMG_SEGMENTATION))
    
    print('\n2. Preprocess dataset to be input for U-Net model')
    print('Getting and resizing train images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        REF_CODE=id_

        if(type(f['#refs#'][id_]) is h5py._hl.dataset.Dataset):
            continue
        else:
            #print(REF_CODE,len(list(f['#refs#'][id_])))
            PICTURE_NAME = f['#refs#'][REF_CODE]['img_name'].value.tostring().decode('utf-8').replace('\x00', '')

            seg=f['#refs#'][REF_CODE]['segmentation'][:]
            cat=f['#refs#'][REF_CODE]['category_label'][:]
            col=f['#refs#'][REF_CODE]['color_label'][:]

            path="./input/image/"
            im = Image.open(path+PICTURE_NAME) 

            im=add_margin(im,0,0,8,0,(0, 0, 0))
            X_train[n] = im

            seg = seg.T

            # Add 8pix at the bottom
            for i in range(8):
                seg=np.r_[seg,[seg[599,:]]]


            seg = seg[:,:,np.newaxis]

            Y_train[n] = seg
            Y_original[n] = seg

            col_train[n] = col
            cat_train[n] = cat

            cat_seg = seg_to_cat(Y_train[n][:,:,0],cat_train[n])       
            for n_cls in range(N_Cls):
                Y_train[n,:,:,n_cls] = np.where(cat_seg == n_cls, 1, 0)

    print('Getting and resizing valid images ... ')
    for n, id_ in tqdm(enumerate(valid_ids), total=len(valid_ids)):
        REF_CODE=id_
        if(type(f['#refs#'][id_]) is h5py._hl.dataset.Dataset):
            continue
        else:
            #print(REF_CODE,len(list(f['#refs#'][id_])))
            PICTURE_NAME = f['#refs#'][REF_CODE]['img_name'].value.tostring().decode('utf-8').replace('\x00', '')

            seg=f['#refs#'][REF_CODE]['segmentation'][:]
            cat=f['#refs#'][REF_CODE]['category_label'][:]
            col=f['#refs#'][REF_CODE]['color_label'][:]

            path="./input/image/"
            im = Image.open(path+PICTURE_NAME) 

            im=add_margin(im,0,0,8,0,(0, 0, 0))
            X_valid[n] = im

            seg = seg.T

            # Add 8pix at the bottom
            for i in range(8):
                seg=np.r_[seg,[seg[599,:]]] #Accumulate row 599 times 8

            #seg = resize(seg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            seg = seg[:,:,np.newaxis]

            Y_valid[n] = seg
            col_valid[n] = col
            cat_valid[n] = cat

            cat_seg = seg_to_cat(Y_valid[n][:,:,0],cat_valid[n])       
            for n_cls in range(N_Cls):
                Y_valid[n,:,:,n_cls] = np.where(cat_seg == n_cls, 1, 0)

    print('Getting and resizing test images ... ')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        REF_CODE=id_
        if(type(f['#refs#'][id_]) is h5py._hl.dataset.Dataset):
            continue
        else:
            PICTURE_NAME = f['#refs#'][REF_CODE]['img_name'].value.tostring().decode('utf-8').replace('\x00', '')

            seg=f['#refs#'][REF_CODE]['segmentation'][:]
            cat=f['#refs#'][REF_CODE]['category_label'][:]
            col=f['#refs#'][REF_CODE]['color_label'][:]

            path="./input/image/"
            im = Image.open(path+PICTURE_NAME) 

            im=add_margin(im,0,0,8,0,(0, 0, 0))
            X_test[n] = im

            seg = seg.T

            # Add 8pix at the bottom
            for i in range(8):
                seg=np.r_[seg,[seg[599,:]]] #Accumulate row 599 times 8

            #seg = resize(seg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            seg = seg[:,:,np.newaxis]

            Y_test[n] = seg
            col_valid[n] = col
            cat_valid[n] = cat

            cat_seg = seg_to_cat(Y_test[n][:,:,0],cat_valid[n])       
            for n_cls in range(N_Cls):
                Y_valid[n,:,:,n_cls] = np.where(cat_seg == n_cls, 1, 0)
    
    # 書き込み
    np.save('./data/X_train.npy', X_train)
    np.save('./data/Y_train.npy', Y_train)
    np.save('./data/col_train.npy', col_train)
    np.save('./data/cat_train.npy', cat_train)

    np.save('./data/X_valid.npy', X_valid)
    np.save('./data/Y_valid.npy', Y_valid)
    np.save('./data/col_valid.npy', col_valid)
    np.save('./data/cat_valid.npy', cat_valid)
    
    np.save('./data/X_test.npy', X_test)
    np.save('./data/Y_test.npy', Y_test)
    np.save('./data/col_test.npy', col_test)
    np.save('./data/cat_test.npy', cat_test)

    print('Done!')
