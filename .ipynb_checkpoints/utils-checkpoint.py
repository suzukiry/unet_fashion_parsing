import os
from datetime import datetime
import json
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
                        verbose=1, 
                        save_best_only=True

                        #monitor='loss',
                        #verbose=0,
                        #save_best_only=False,
                        #save_weights_only=True,
                        #mode='auto',
                        #period=5
    )
    
    csv_filepath = os.path.join(model_dir, 'loss.csv')
    csv = CSVLogger(csv_filepath, append=True)


    #return cp, csv
    return earlystopper, cp
    