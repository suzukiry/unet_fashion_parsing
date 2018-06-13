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
import pandas as pd
import matplotlib.pyplot as plt

# Train : Valid : Test = 0.72 : 0.08 : 0.2
TRAIN_RATIO=0.9 * 0.9 #0.01#
VALID_RATIO=0.9 * 0.1 #0.01#
TEST_RATIO=0.1 #test #0.01

# Set some parameters
IMG_WIDTH = 400
IMG_HEIGHT = 608 #608#592
IMG_CHANNELS = 3
IMG_SEGMENTATION = 425
N_Cls=23

model_dir = ""

filepath = './input/fashon_parsing_data.mat'
arrays = {}
f = h5py.File(filepath)

def adjust_ax(df, ax, ylabel):
    df.plot(ax=ax)
    ax.set_title(ylabel)
    ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

def plot_learningcurve(df_history):
    figsize = (12, 4)
    nrows = 1
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    #for ax, lbl in zip(axes.ravel(), ('acc', 'loss')):
    for ax, lbl in zip(axes.ravel(), ('loss', 'mean_iou')):
        df = df_history[[lbl, 'val_{}'.format(lbl)]]
        ax = adjust_ax(df, ax, ylabel=lbl)

    plot_filepath = os.path.join(model_dir, 'learning_curve.png')
    plt.savefig( plot_filepath )
    
def plot_learningcurve_from_csv(csv_filepath):
    df_history = pd.read_csv(csv_filepath)
    plot_learningcurve(df_history)


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

def batch_iter(batch_size = 16, shuffle=True, mode=None):

    ids = fashon_parsing_data_ids()

    train_data_count=int(len(ids) * TRAIN_RATIO)
    valid_data_count=int(len(ids) * VALID_RATIO)
    test_data_count=int(len(ids) * TEST_RATIO)
    #print(train_data_count,valid_data_count,test_data_count)

    if mode=="train":
        train_ids=ids[:train_data_count]
        
        steps_per_epoch = math.ceil(len(train_ids)/batch_size) #round        
        gen_ids = train_ids

    elif mode=="valid":
        valid_ids=ids[train_data_count:train_data_count+valid_data_count]

        steps_per_epoch = math.ceil(len(valid_ids)/batch_size) #round
        gen_ids = valid_ids
 
    elif mode=="test":
        test_ids=ids[train_data_count+valid_data_count:]

        steps_per_epoch = math.ceil(len(test_ids)/batch_size) #round
        gen_ids = test_ids

    #print(len(gen_ids))
    
    def data_generator():
        #print('generator initiated')
        count = 0
                
        while True:
            # Shuffle the data at each epoch            
            if shuffle:
                n_data = len(gen_ids)
                indices = np.arange(n_data)
                np.random.shuffle(indices)

                gen_ids_shuffled = gen_ids[indices]

            else:
                continue

            for i in range(0, n_data, batch_size):

                batch_ids = gen_ids_shuffled[i: i + batch_size]

                X_batch = np.zeros((len(batch_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
                Y_batch = np.zeros((len(batch_ids), IMG_HEIGHT, IMG_WIDTH, N_Cls))
                col_batch = np.zeros((len(batch_ids), 1, IMG_SEGMENTATION))
                cat_batch = np.zeros((len(batch_ids), 1, IMG_SEGMENTATION))

                for n, id_ in enumerate(batch_ids):
                    REF_CODE=id_

                    PICTURE_NAME = f['#refs#'][REF_CODE]['img_name'].value.tostring().decode('utf-8').replace('\x00', '')

                    seg=f['#refs#'][REF_CODE]['segmentation'][:]
                    cat=f['#refs#'][REF_CODE]['category_label'][:]
                    col=f['#refs#'][REF_CODE]['color_label'][:]

                    path="./input/image/"
                    im = Image.open(path+PICTURE_NAME) 

                    im=add_margin(im,0,0,8,0,(0, 0, 0))
                    X_batch[n] = im

                    seg = seg.T

                    # Add 8pix at the bottom
                    for i in range(8):
                        seg=np.r_[seg,[seg[599,:]]]

                    seg = seg[:,:,np.newaxis]

                    Y_batch[n] = seg

                    col_batch[n] = col
                    cat_batch[n] = cat

                    cat_seg = seg_to_cat(Y_batch[n][:,:,0],cat_batch[n])       
                    for n_cls in range(N_Cls):
                        Y_batch[n,:,:,n_cls] = np.where(cat_seg == n_cls, 1, 0)

                yield X_batch, Y_batch
                #print('generator yielded a batch {} - {},{}'.format(count, X_batch.shape, Y_batch.shape))
                #count += 1

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

def file_path(file=None):
    if file == "csv":
        return os.path.join(model_dir, 'loss.csv')

def ready_fitting(model):

    global model_dir
    
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

    print('Ready model.json: ', model_dir)

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

    print("Ready csv_filepath: {}".format(csv_filepath))
    return cp, csv
    #return earlystopper, cp

    
