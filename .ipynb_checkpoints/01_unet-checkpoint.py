from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
import numpy as np

from utils import *
import math

# Fit model    
EPOCHS = 25

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def get_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_Cls):
     
    # Build U-Net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs) #0〜1の範囲に正規化

    # Contracting 1
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    # Contracting 2
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    # Contracting 3
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    # Contracting 4
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    # Lowest resolution
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    # Up-sampling 1
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    # Up-sampling 2
    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    # Up-sampling 3
    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    # Up-sampling 4
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(N_Cls, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    
    return model

if __name__ == '__main__':

    # Set parameters
    IMG_WIDTH = 400
    IMG_HEIGHT = 608
    IMG_CHANNELS = 3
    IMG_SEGMENTATION = 425
    N_Cls = 23

    BATCH_SIZE = 16
   
    #X_train = np.load('./data/X_train.npy')
    #Y_train = np.load('./data/Y_train.npy')

    print("\n1. Create U-Net")
    model = get_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_Cls)    
    print("Done!")



    #earlystopper, checkpointer = ready_fitting(model)
    cp, csv = ready_fitting(model)

    print("\n2. Fit U-Net model")
    
    train_steps, train_batches = batch_iter(BATCH_SIZE,mode="train")
    valid_steps, valid_batches = batch_iter(BATCH_SIZE,mode="valid")

    results = model.fit_generator(
        train_batches,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=valid_batches,
        validation_steps=valid_steps,
        callbacks=[cp, csv]
    )

    plot_learningcurve_from_csv(file_path(file="csv"))
    print("Done!")
    
