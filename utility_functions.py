#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
from scipy import io
from Feature_Extractor import *
import pandas as pd

LABELS = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
NUM_CHANNELS = {
    'Dog_1': 16,
    'Dog_2': 16,
    'Dog_3': 16,
    'Dog_4': 16,
    'Dog_5': 15,
    'Patient_1': 15,
    'Patient_2': 24,        
}

WINDOW = 20
WINDOW_SHIFT = 10

def read_data(filename):
    """Convert .mat data from 'filename' into a dict 'processed_data'."""
    try:
        mat_data = io.loadmat(filename)
    except TypeError: 
        raise FileNotFoundError
    for k in mat_data.keys():
        if 'segment' in k:
            key = k
    rawdata = mat_data[key]
    data = rawdata['data'][0, 0]
    ch_data = rawdata['channels'][0, 0]
    channels = np.array([ch_data[0, i][0] for i in range(data.shape[0])])
    sampling_frequency = rawdata['sampling_frequency'][0, 0][0, 0]
    data_length_sec = rawdata['data_length_sec'][0, 0][0, 0]
    try:
        sequence = rawdata['sequence'][0, 0][0, 0]
    except: 
        sequence = None
    processed_data = {
        'data': data,
        'channels': channels,
        'sampling_frequency': sampling_frequency,
        'data_length_sec': data_length_sec,
        'sequence': sequence
    }
    return processed_data


def extract_features_from_file(features, filename):
    """Extract data of specific 'feature' from 'filename'."""
    data = read_data(filename)
    time_series = data['data']
    freq = data['sampling_frequency']
    windowed_data = {
        'data': None,
        'sampling_frequency': freq,
        'sequence': data['sequence'],
        'channels': data['channels']
    }
    T = 0
    freq_floor = np.floor(freq)
    num_windows = int((data['data_length_sec'] - WINDOW) / WINDOW_SHIFT) + 1
    res = {} 
    for i in range(num_windows):
        start, end = int(T * freq_floor), int((T + WINDOW) * freq_floor)
        windowed_data['data'] = time_series[:, start:end]
        fe = Feature_Extractor(windowed_data)
        T += WINDOW_SHIFT
        for feature in features:
            method = getattr(fe, 'extract_' + feature)
            res[feature] = method() if feature not in res.keys() else np.vstack(
                    [res[feature], method()])
    return res

def process_features_to_csv(
        features, dirname, prefix, dest_folder=None, type_list=None):
    """Process time-series files in folder 'dirname' of file types
    specified in 'type_list', and extract data of 'feature'.
    Save the result as a csv file in 'dest_folder'.
    Possible type_list choices: 'interictal', 'preictal', 'test'.
    """ 
    if type_list is None:
        type_list = ['interictal', 'preictal', 'test']

    for t in type_list:
        print('start processing {} files...'.format(t))
        seg_no = 0
        fdata = {}
        row_index = []
        end = False
        while not end:
            seg_no += 1
            nstr = str(seg_no).rjust(4, '0')
            filename = prefix + '_' + t + '_segment_' + nstr + '.mat'
            src_filename = dirname + '/' + filename
            try:
                print(src_filename)
                feature_dict = extract_features_from_file(features, src_filename)
                num_windows = feature_dict[features[0]].shape[0]
                row_index += [filename] * num_windows
                for feature in features:
                    fdata[feature] = np.vstack(
                        [fdata[feature], feature_dict[feature]]
                    ) if feature in fdata.keys() else feature_dict[feature]
            except FileNotFoundError:
                print('{} {} files processed'.format(seg_no - 1, t))
                end = True
        dest_dir = dest_folder + '/' if dest_folder else './'
        for feature in features:
            dest_filename = dest_dir + prefix + '_' + feature + '_' + t + '.csv'
            cols = getattr(Feature_Extractor, feature + '_feature_names')(
                    NUM_CHANNELS[prefix])
            print(fdata[feature].shape, cols)
            df = pd.DataFrame(
                    fdata[feature], columns=cols)
            df.index = row_index
            df.to_csv(dest_filename, index=True)
    return dest_dir   

def generate_result_csv(result_dict_by_label, dest_filename):
    need_header = True
    for label in LABELS:
        print(label)
        try:
            res = result_dict_by_label[label]
            clip_names = ['{}_test_segment_{}.mat'.format(
                label, str(i + 1).rjust(4, '0')) for i in range(len(res))]
            x = np.vstack([np.array(clip_names), res]).T
            df = pd.DataFrame(x, columns=['clip', 'preictal'])
            if need_header:
                with open(dest_filename, 'w') as f:
                    df.to_csv(f, header=need_header, index=False)
            else:
                with open(dest_filename, 'a') as f:
                    df.to_csv(f, header=need_header, index=False)
            need_header = False
        except KeyError:
            pass
        
