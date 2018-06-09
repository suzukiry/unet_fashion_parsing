import os
from datetime import datetime
import json
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image
import sys, math

# Train : Valid : Test = 0.72 : 0.08 : 0.2
TRAIN_RATIO=0.01#0.8 * 0.9
VALID_RATIO=0.01#0.8 * 0.1
TEST_RATIO=0.01 #test

# Set some parameters
IMG_WIDTH = 400
IMG_HEIGHT = 608 #608#592
IMG_CHANNELS = 3
IMG_SEGMENTATION = 425
N_Cls=23

filepath = './input/fashon_parsing_data.mat'
arrays = {}
f = h5py.File(filepath)


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


def fashon_parsing_data_ids():
    update_ids = []

    for k, v in f.items():
        arrays[k] = np.array(v)
    
    # Get train and test IDs
    ids = list(arrays['#refs#'])
    
    for id_ in ids:
        
        if(type(f['#refs#'][id_]) is h5py._hl.dataset.Dataset):
            continue
        else:
            update_ids.append(id_)

    return np.array(update_ids)

def batch_iter(batch_size = 16, shuffle=True):

    ids = fashon_parsing_data_ids()

    train_data_count=int(len(ids) * TRAIN_RATIO)
    train_ids=ids[:train_data_count]
    
    steps_per_epoch = math.ceil(len(train_ids)/batch_size) #round
    
    def data_generator():
        print('generator initiated')
        count = 0
        
        while True:
            # Shuffle the data at each epoch
            
            if shuffle:

                n_data = len(train_ids)
                train_indices = np.arange(n_data)
                np.random.shuffle(train_indices)

                #print(n_data)
                train_ids_shuffled = train_ids[train_indices]
                train_ids_input = train_ids_shuffled

            else:
                continue

            for i in range(0, n_data, batch_size):
            #for i in range(steps_per_epoch):

                #start_index = i * batch_size # Start with 0
                #end_index = min((i + 1) * batch_size, n_data)
                #train_batch_ids = train_ids_input[start_index: end_index]

                train_batch_ids = train_ids_input[i: i + batch_size]


                X_train = np.zeros((len(train_batch_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
                Y_train = np.zeros((len(train_batch_ids), IMG_HEIGHT, IMG_WIDTH, N_Cls))
                col_train = np.zeros((len(train_batch_ids), 1, IMG_SEGMENTATION))
                cat_train = np.zeros((len(train_batch_ids), 1, IMG_SEGMENTATION))

                for n, id_ in enumerate(train_batch_ids):
                    REF_CODE=id_

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

                    col_train[n] = col
                    cat_train[n] = cat

                    cat_seg = seg_to_cat(Y_train[n][:,:,0],cat_train[n])       
                    for n_cls in range(N_Cls):
                        Y_train[n,:,:,n_cls] = np.where(cat_seg == n_cls, 1, 0)

                yield X_train, Y_train
                #yield train_batch_ids
                print('generator yielded a batch {} - {},{}'.format(count, X_train.shape, Y_train.shape))
                #print('generator yielded a batch {}'.format(count))
                #print('----------------')
                count += 1

    return steps_per_epoch, data_generator()

# NOT USED
def get_batches(x, y, batch_size): 

    n_data = len(x)
    indices = np.arange(n_data)
    np.random.shuffle(indices)
    
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    for i in range(0, n_data, batch_size):
        x_batch = x_shuffled[i: i + batch_size]
        y_batch = y_shuffled[i: i + batch_size]    
        yield x_batch, y_batch

def ready_fitting(model):

    print('\nReady fitting ... ')
    model_dir = os.path.join(
        'models',
        datetime.now().strftime('%y%m%d_%H%M')
    )

    os.makedirs(model_dir,exist_ok=True)
    print('Ready model_dir: ', model_dir)

    dir_weights = os.path.join(model_dir, 'weights')
    os.makedirs(dir_weights, exist_ok=True)
    print('Ready dir_weights: ', dir_weights)
    
    model_json = os.path.join(model_dir, 'model.json')
    with open(model_json, 'w') as f:
        json.dump(model.to_json(), f)

    print('Ready model.json', model_dir)

    cp_filepath = os.path.join(dir_weights, 'ep_{epoch:02d}_ls_{loss:.1f}.h5')
    print('Ready cp_filepath: ', dir_weights)
    
    earlystopper = EarlyStopping(patience=5, verbose=1)
    
    cp = ModelCheckpoint(
                        cp_filepath, 
                        #verbose=1, 
                        #save_best_only=True

                        monitor='loss',
                        verbose=0,
                        save_best_only=False,
                        save_weights_only=True,
                        mode='auto',
                        period=2
    )
    
    csv_filepath = os.path.join(model_dir, 'loss.csv')
    csv = CSVLogger(csv_filepath, append=True)

    return cp, csv
    #return earlystopper, cp

    
