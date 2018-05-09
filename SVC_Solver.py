#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import constant
from sklearn import preprocessing

PREICTAL = 1
INTERICTAL = 0
# =============================================================================
# feature_names = ['band_power']
# =============================================================================

class SVC_Solver:
    
    def __init__(
            self, feature_names, C=1.0, kernel='rbf', gamma=1e-4, probability=True
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.feature_names = feature_names
        self.probability = probability
        self.clf = SVC(
            C=C, kernel=kernel, gamma=gamma, probability=probability)
        
    def _get_data_by_label_type(self, label, typ):
        d = {}
        for name in self.feature_names:
            filename = label + '_' + name + '_' + typ + '.csv'
            d[name] = pd.read_csv(constant.FEATURE_DEST_PATH + '/' + filename)
        d = self._preproc_feature_data(d)
        return d
    
    def _preproc_feature_data(self, raw_feature_data):
# =============================================================================
#         for name in raw_feature_data.keys():
#             X = raw_feature_data[name]
#             raw_feature_data[name] = preprocessing.scale(X)
# =============================================================================
        return raw_feature_data

    def fit_by_label(self, label):
        pre_data = self._get_data_by_label_type(label, 'preictal')
        inter_data = self._get_data_by_label_type(label, 'interictal')
        X_train_pre, X_train_inter = None, None
        for feature in self.feature_names:
            temp = pre_data[feature]
            X_train_pre = temp if X_train_pre is None else np.hstack([X_train_pre, temp])
            temp = inter_data[feature]
            X_train_inter = temp if X_train_inter is None else np.hstack([X_train_inter, temp])
        X_train = np.vstack([X_train_pre, X_train_inter])
        y_train = np.concatenate([
            np.array([PREICTAL] * X_train_pre.shape[0]),
            np.array([INTERICTAL] * X_train_inter.shape[0])
        ])
        self.clf.fit(X_train, y_train)

    def predict_by_label(self, label):
        test_data = self._get_data_by_label_type(label, 'test')
        X_test = None
        for feature in self.feature_names:
            temp = test_data[feature]
            X_test = temp if X_test is None else np.hstack([X_test, temp])
        if self.probability:
            res = self.clf.predict_proba(X_test)[:, 1]
        else:
            res = self.clf.predict(X_test)
        return res
    