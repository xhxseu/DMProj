#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import preprocessing

DATA_FILE_PATH = r'.'

# FOLDER where you store your feature data
FEATURE_DEST_PATH = r'./processedfeature_Dog_3'

PREICTAL = 1
INTERICTAL = 0

WINDOW = 20 
WINDOW_SHIFT = 10

def get_data_by_label_type(label, typ, feature_names):
    d = {}
    for name in feature_names:
        filename = label + '_' + name + '_' + typ + '.csv'
        temp = pd.read_csv(FEATURE_DEST_PATH + '/' + filename)
        
        temp = temp.T.fillna(temp.mean(axis=1)).T
        temp = np.array(temp)[:, 1:].astype(float)
#        temp = pd.DataFrame(temp).apply(np.log)
        print(temp.shape)
        d[name] = temp
    full_data = None    
    for name in feature_names: 
        full_data = np.array(d[name]) if full_data is None else np.hstack(
                [full_data, np.array(d[name])])
        
    return full_data

def generate_fit_data(label, features, samp, r=1, cv=False):
    pre = get_data_by_label_type(label, 'preictal', features)
    inter = get_data_by_label_type(label, 'interictal', features)
    
    num = int((600 - WINDOW) / WINDOW_SHIFT + 1) * 6
    num_of_file = int(pre.shape[0] / num)
    num_test = int(num_of_file * 0.34)
    t = np.random.choice(num_of_file, num_test, replace=False)
    pre_train, pre_test = None, None
    for i in range(num_of_file):
        start, end = i * num, (i + 1) * num 
        temp_pre = pre[start:end, :] 
        if i in t:
            pre_test = temp_pre if pre_test is None else np.vstack((pre_test, temp_pre))
        else:
            pre_train = temp_pre if pre_train is None else np.vstack((pre_train, temp_pre))
            
    num_of_file_inter = int(inter.shape[0] / num)
    ratio = int(num_of_file_inter / num_of_file / r)
    num_of_file = r * num_of_file if num_of_file_inter > r * num_of_file else num_of_file_inter
    num_test = int(num_of_file * 0.34)
    print(num_test)
    t = np.random.choice(num_of_file, num_test, replace=False)
    inter_train, inter_test = None, None
    for j in range(num_of_file):
        i = j * ratio
        start, end = i * num, (i + 1) * num 
        temp_inter = inter[start:end, :]
        if j in t:
            inter_test = temp_inter if inter_test is None else np.vstack((inter_test, temp_inter))
        else:
            inter_train = temp_inter if inter_train is None else np.vstack((inter_train, temp_inter))
        
    print(inter_test.shape, inter_train.shape, inter.shape)

    X_train = np.vstack([pre_train, inter_train])
    y_train = np.concatenate([
        np.array([PREICTAL] * pre_train.shape[0]),
        np.array([INTERICTAL] * inter_train.shape[0])
    ])    
    X_test = np.vstack([pre_test, inter_test])
    y_test = np.concatenate([
        np.array([PREICTAL] * pre_test.shape[0]),
        np.array([INTERICTAL] * inter_test.shape[0])
    ])
# =============================================================================
#     while inter.shape[0] > 5.5 * pre.shape[0]:
#         inter = inter[np.random.choice(inter.shape[0], int(pre.shape[0] * 5), replace=False)]
#         
#     print(inter.shape, pre.shape)
#     
#     full_X = np.vstack([pre, inter])
#     full_y = np.concatenate([
#         np.array([PREICTAL] * pre.shape[0]),
#         np.array([INTERICTAL] * inter.shape[0])
#     ])
# =============================================================================
    
    if cv:
#        X_train, X_test, y_train, y_test = train_test_split(
#            full_X, full_y, test_size=0.33, random_state=42)
    
        X_train = X_train[np.array(range(0, y_train.shape[0], samp))]
        y_train = y_train[np.array(range(0, y_train.shape[0], samp))]
        
        X_test = X_test[np.array(range(0, y_test.shape[0], samp))]
        y_test = y_test[np.array(range(0, y_test.shape[0], samp))]
        
    else:
        full_X = np.vstack([X_train, X_test])
        full_y = np.vstack([y_train, y_test])

        X_train = full_X[np.array(range(0, full_y.shape[0], samp))]
        y_train = full_y[np.array(range(0, full_y.shape[0], samp))]
        X_test = None,
        y_test = None

    return X_train, X_test, y_train, y_test 

    
def generate_fit_data_tuh(label, features, samp, cv=True):
        
        for i,fea in enumerate(features): 
            X_train_fea = pd.read_csv('./tuh_processedfeature/TUH_'+fea+'_train.csv')
            X_train_fea = np.array(X_train_fea)[:, 1:].astype(float)
            X_test_fea = pd.read_csv('./tuh_processedfeature/TUH_'+fea+'_test.csv')
            X_test_fea = np.array(X_test_fea)[:, 1:].astype(float)
            
            if i==0: 
                X_train = X_train_fea;
                X_test = X_test_fea;
            else:
                X_train = np.hstack((X_train,X_train_fea))
                X_test = np.hstack((X_test,X_test_fea))

        y_train = np.load('./tuh_processedfeature/TUH_y_train.npy')
        y_test = np.load('./tuh_processedfeature/TUH_y_test.npy') 
        
        return X_train, X_test, y_train, y_test

def fit_by_label_gs(label, feature, samp, sca=True, hypers=None):
    if label == 'TUH':
        X_train, X_test, y_train, y_test = generate_fit_data_tuh(label, feature, samp, cv=True)
    else:
        X_train, X_test, y_train, y_test = generate_fit_data(label, feature, samp, cv=True)
    scaler = preprocessing.StandardScaler().fit(X_train)
    (X_train, X_test) = (scaler.transform(X_train), scaler.transform(
    X_test)) if sca is True else (X_train, X_test)

    svc = SVC(C=1.0, kernel='rbf')
    if hypers is None:
        kernels = ['rbf']
        Cs = 10**np.linspace(-2, 2, 10)
        gammas = 10**np.linspace(-4, -2, 10)
        tuned_parameters = [{'kernel': kernels, 'C': Cs, 'gamma': gammas}]
    else:
        tuned_parameters = hypers
    
    n_folds = 5
    gs = GridSearchCV(svc, tuned_parameters, cv=n_folds, scoring='roc_auc')
    gs.fit(X_train, y_train)
    print(gs.score(X_test, y_test)) 
    print(gs.best_estimator_) 
    return gs 

def fit_by_label(label, feature, samp, params):
    X_train, X_test, y_train, y_test = generate_fit_data(label, feature, samp)
    gamma, C, kernel = params['gamma'], params['C'], params['kernel']
    svm = SVC(gamma=gamma, C=C, kernel=kernel, probability=True)
    svm.fit(X_train, y_train)
    return svm 

def predict_by_label(label, features, clf):
    test_data = get_data_by_label_type(label, 'test', features)
#    test_data = np.array(test_data)[:, 1:].astype(float)
    num = int((600 - WINDOW) / WINDOW_SHIFT + 1)
    num_of_file = int(test_data.shape[0] / num)
    
    y_pred = np.zeros(num_of_file)
    res = clf.predict_proba(test_data)[:, 1]
    for i in range(num_of_file):
        start, end = (i) * num, (i + 1) * num
        prob_data = res[start:end]
#        print(prob_data)
#        y_pred[i] = np.nanmedian(prob_data)
        y_pred[i] = len(prob_data[prob_data > 0.5]) / len(prob_data)

    print(y_pred.shape)
    return y_pred

#%% svm for kaggle data

la = 'Dog_3'
fea = ['IncAccEnergy']

hypers = {
    'C': 10**np.linspace(-2, 2, 10),
    'gamma': 10**np.linspace(-4, 2, 10), 
    'kernel': ['rbf']
} 
np.random.seed(1)  
gs = fit_by_label_gs(la, fea, 2, sca=True, hypers=hypers)
params = {
    'C': gs.best_params_['C'], 
    'gamma': gs.best_params_['gamma'], 
    'kernel': gs.best_params_['kernel'] 
} 

np.save('./processedfeature/parameter_results/svm_params_'+fea[0]+'.npy',params)

#%%
np.random.seed(1) 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 
X_train, X_test, y_train, y_test = generate_fit_data(la, fea, 1, r=2, cv=True)
gamma, C, kernel = params['gamma'], params['C'], params['kernel']
svm = SVC(gamma=gamma, C=C, kernel=kernel, probability=True) 
svm.fit(X_train, y_train)
print(X_train.shape, X_test.shape) 
#svm.predict_proba(X_test)[:, 1] 
r = svm.predict(X_test)  
print(confusion_matrix(r, y_test))  

y_score = svm.predict_log_proba(X_test)[:, 1]
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, y_score) 
svm_auc = roc_auc_score(y_test, y_score) 
score = svm.score(X_test, y_test) 

np.save('./processedfeature/parameter_results/svm_fpr_'+fea[0]+'.npy',svm_fpr)
np.save('./processedfeature/parameter_results/svm_tpr_'+fea[0]+'.npy',svm_tpr)
np.save('./processedfeature/parameter_results/svm_auc_'+fea[0]+'.npy',svm_auc)

fig = plt.figure() 
plt.plot(svm_fpr,svm_tpr,'r',label=fea[0]+'(area={:.3f})'.format(svm_auc))
plt.legend() 
plt.xlabel('FPR')  
plt.ylabel('TPR') 
plt.title('ROC') 
plt.show() 

score  
svm_auc 

#%% logistic regression for kaggle data
la = 'Dog_3'
fea = ['hurst'] 

np.random.seed(1) 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 
X_train, X_test, y_train, y_test = generate_fit_data(la, fea, 1, r=2, cv=True)
lg = LogisticRegressionCV()  
lg.fit(X_train, y_train) 
print(X_train.shape, X_test.shape) 
r = lg.predict(X_test) 
print(confusion_matrix(r, y_test))  

y_score = lg.predict_log_proba(X_test)[:, 1]
lg_fpr, lg_tpr, lg_thresholds = roc_curve(y_test, y_score) 
lg_auc = roc_auc_score(y_test, y_score) 
score = lg.score(X_test, y_test) 

np.save('./processedfeature_Dog_3/parameter_results/lg_fpr_'+fea[0]+'.npy',lg_fpr)
np.save('./processedfeature_Dog_3/parameter_results/lg_tpr_'+fea[0]+'.npy',lg_tpr)
np.save('./processedfeature_Dog_3/parameter_results/lg_auc_'+fea[0]+'.npy',lg_auc)

fig = plt.figure() 
plt.plot(lg_fpr,lg_tpr,'r',label=fea[0]+'(area={:.3f})'.format(lg_auc))
plt.legend() 
plt.xlabel('FPR')  
plt.ylabel('TPR') 
plt.title('ROC') 
plt.show() 
          
score  
lg_auc


#%% svm for tuh data

la = 'TUH'
fea = ['band_power'] 

hypers = {
    'C': 10**np.linspace(-2, 2, 10),
    'gamma': 10**np.linspace(-4, 2, 10), 
    'kernel': ['rbf']
} 
np.random.seed(1)  
gs = fit_by_label_gs(la, fea, 2, sca=True, hypers=hypers)
params = {
    'C': gs.best_params_['C'], 
    'gamma': gs.best_params_['gamma'], 
    'kernel': gs.best_params_['kernel'] 
} 

np.save('./tuh_processedfeature/tuh_parameter_results/svm_params_'+fea[0]+'.npy',params)

#%% svm for tuh data  
np.random.seed(1) 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = generate_fit_data_tuh(la, fea, 1, cv=True)
gamma, C, kernel = params['gamma'], params['C'], params['kernel']
svm = SVC(gamma=gamma, C=C, kernel=kernel, probability=True) 
svm.fit(X_train, y_train)
print(X_train.shape, X_test.shape) 
#svm.predict_proba(X_test)[:, 1]   
r = svm.predict(X_test)   
print(confusion_matrix(r, y_test))  

y_score = svm.predict_log_proba(X_test)[:, 1]
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, y_score) 
svm_auc = roc_auc_score(y_test, y_score) 
score = svm.score(X_test, y_test) 

np.save('./tuh_processedfeature/tuh_parameter_results/svm_fpr_'+fea[0]+'.npy',svm_fpr)
np.save('./tuh_processedfeature/tuh_parameter_results/svm_tpr_'+fea[0]+'.npy',svm_tpr)
np.save('./tuh_processedfeature/tuh_parameter_results/svm_auc_'+fea[0]+'.npy',svm_auc)

fig = plt.figure() 
plt.plot(svm_fpr,svm_tpr,'r',label=fea[0]+'(area={:.3f})'.format(svm_auc))
plt.legend() 
plt.xlabel('FPR')  
plt.ylabel('TPR') 
plt.title('ROC') 
plt.show() 

score  
svm_auc 


#%% logistic regression for tuh data
la = 'TUH'
fea = ['moment_3rd']

np.random.seed(1) 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 
X_train, X_test, y_train, y_test = generate_fit_data_tuh(la, fea, 1, cv=True)
lg = LogisticRegressionCV()  
lg.fit(X_train, y_train) 
print(X_train.shape, X_test.shape) 
r = lg.predict(X_test) 
print(confusion_matrix(r, y_test))  

y_score = lg.predict_proba(X_test)[:, 1]
lg_fpr, lg_tpr, lg_thresholds = roc_curve(y_test, y_score) 
lg_auc = roc_auc_score(y_test, y_score) 
score = lg.score(X_test, y_test) 

np.save('./processedfeature/parameter_results/lg_fpr_'+fea[0]+'.npy',lg_fpr)
np.save('./processedfeature/parameter_results/lg_tpr_'+fea[0]+'.npy',lg_tpr)
np.save('./processedfeature/parameter_results/lg_auc_'+fea[0]+'.npy',lg_auc)

fig = plt.figure() 
plt.plot(lg_fpr,lg_tpr,'r',label=fea[0]+'(area={:.3f})'.format(lg_auc))
plt.legend() 
plt.xlabel('FPR')  
plt.ylabel('TPR') 
plt.title('ROC') 
plt.show() 
          
score  
lg_auc
#%%
from utility_functions import generate_result_csv
res = {}
#svm = fit_by_label(la, fea, 10, params)
res[la] = predict_by_label(la, fea, svm)
generate_result_csv(res, 'result_d5.csv')

#%%
clf = SVC(kernel='linear')
selector = RFECV(clf, 2)
selector.fit(X_train, y_train)

#%%
X_train_red = X_train[:, selector.support_]
X_test_red = X_test[:, selector.support_]
clf = SVC(kernel='rbf', gamma=gs.best_params_['gamma'], C=gs.best_params_['C'])
clf.fit(X_train_red, y_train)
clf.score(X_test_red, y_test)

#%%
#la1 = 'Dog_4'
#fea1 = ['band_power']
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.linear_model import LogisticRegressionCV
#X_train, X_test, y_train, y_test = generate_fit_data(la1, fea1, 1, cv=True)
#
#clf = LinearDiscriminantAnalysis()
#clf.fit(X_train, y_train) 
#
#r = clf.predict(X_test)
#print(confusion_matrix(r, y_test), clf.score(X_test, y_test))
#
#clf1 = LogisticRegressionCV(penalty='l2', scoring='roc_auc')
#clf1.fit(X_train, y_train)
# 
#r = clf1.predict(X_test)  
#print(confusion_matrix(r, y_test), clf1.score(X_test, y_test))
#%%
for string in zip(['d1', 'd2', 'd3', 'd4', 'd5', 'p1', 'p2'], LABELS):
    a = pd.read_csv('result_' + string[0] + '.csv')
    res[string[1]] = np.array(a)[:, 1]
generate_result_csv(res, 'result.csv')