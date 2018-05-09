# -*- coding: utf-8 -*-
 
#%%
import pickle
import numpy as np
import pandas as pd
import sklearn

import pickle
import numpy as np
import pandas
import sklearn

#train_data = pickle.load(open('./tuh_rawdata/TUH_train_60s_window_10s_preictal.p', 'rb'))
#seed = 7
#np.random.seed(seed)
#X_train = train_data['X']
#y_train = np.array(train_data['Y'])
#X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=0)
#
#test_data = pickle.load(open('./tuh_rawdata/TUH_test_60s_window_10s_preictal.p', 'rb'))
#seed = 7
#np.random.seed(seed)
#X_test = test_data['X'] 
#y_test = np.array(test_data['Y'])
#X_test, y_test = sklearn.utils.shuffle(X_test, y_test, random_state=0)
#
#np.save('./tuh_processedfeature/TUH_X_train.npy', X_train) 
#np.save('./tuh_processedfeature/TUH_y_train.npy', y_train) 
#np.save('./tuh_processedfeature/TUH_X_test.npy', X_test) 
#np.save('./tuh_processedfeature/TUH_y_test.npy', y_test) 

X_train = np.load('./tuh_processedfeature/TUH_X_train.npy')
X_test = np.load('./tuh_processedfeature/TUH_X_test.npy')
y_train = np.load('./tuh_processedfeature/TUH_y_train.npy')
y_test = np.load('./tuh_processedfeature/TUH_y_test.npy') 
#%% 

def extract_features(features, X_data):
    """Extract data of specific 'feature' from 'filename'."""
    time_series = X_data
    freq = 200
    processed_data = {
        'data': time_series[:, :], 
        'sampling_frequency': freq, 
        'channels': np.arange(19)
    }
    res = {} 
    for i in range(1):
        fe = Feature_Extractor(processed_data)
        for feature in features: 
            method = getattr(fe, 'extract_' + feature)
            res[feature] = method() if feature not in res.keys() else np.vstack(
                    [res[feature], method()])
    return res 
#%%
from Feature_Extractor import Feature_Extractor
def process_features_to_csv(
        X_typ, features, dest_folder=None):
    """Process time-series files in folder 'dirname' of file types
    specified in 'type_list', and extract data of 'feature'.
    Save the result as a csv file in 'dest_folder'.
    Possible type_list choices: 'interictal', 'preictal', 'test'.
    """ 
    X = X_train if X_typ is 'train' else X_test
    dest_dir = dest_folder + '/' if dest_folder else './'
    fdata = {}
    for i in range(X.shape[0]): 
        xdata = X[i].T 
        res = extract_features(features, xdata)
        for feature in features: 
            fdata[feature] = np.vstack( 
                [fdata[feature], res[feature]]
                ) if feature in fdata.keys() else res[feature]
    for feature in features:
        dest_filename = dest_dir + 'TUH' + '_' + feature + '_' + X_typ + '.csv'
        cols = getattr(Feature_Extractor, feature + '_feature_names')(19)
#            print(fdata[feature].shape, cols)
    df = pd.DataFrame(fdata[feature], columns=cols)
    df.to_csv(dest_filename, index=True)
    return dest_dir 

#%%
process_features_to_csv('train',['band_power'],'./tuh_processedfeature_montage') 






