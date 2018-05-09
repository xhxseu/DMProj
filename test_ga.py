# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:45:33 2018

@author: Jianhan Song
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from feature_selection_ga import FeatureSelectionGA
from sklearn.ensemble import RandomForestClassifier


PREICTAL = 1
INTERICTAL = 0

# FOLDER of Data, e.g. Dog_1 data should be in DATA_FILE_PATH + '/Dog_1'

# FOLDER where you store your feature data
FEATURE_DEST_PATH = '~/Documents/DMProj/Feature20_ds'

#%%
def get_data_by_label_type(label, typ, feature_names):
    d = {} 
    for name in feature_names:
        filename = label + '_' + name + '_' + typ + '.csv'
#        temp = pd.read_csv(constant.FEATURE_DEST_PATH + '/' + filename)
        temp = pd.read_csv(FEATURE_DEST_PATH + '/' + filename)
        temp = temp.T.fillna(temp.mean(axis=1)).T
        temp = np.array(temp)[:, 1:].astype(float)
#        temp = pd.DataFrame(temp).apply(np.log)

        d[name] = temp
    full_data = None    
    for name in feature_names:
        full_data = np.array(d[name]) if full_data is None else np.hstack(
                [full_data, np.array(d[name])])
    return full_data

def generate_fit_data(label, features, samp, r=1, cv=False):
    pre = get_data_by_label_type(label, 'preictal', features)
    inter = get_data_by_label_type(label, 'interictal', features)
    
    num = int((600 - 20) / 10 + 1) * 6
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
    num_of_file = r * num_of_file if num_of_file_inter >= r * num_of_file else num_of_file_inter
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
            X_train_fea = pd.read_csv('~/Documents/DMProj/tuh_processedfeature/TUH_'+fea+'_train.csv')
            X_train_fea = np.array(X_train_fea)[:, 1:].astype(float)
            X_test_fea = pd.read_csv('~/Documents/DMProj/tuh_processedfeature/TUH_'+fea+'_test.csv')
            X_test_fea = np.array(X_test_fea)[:, 1:].astype(float)
            
            if i==0: 
                X_train = X_train_fea;
                X_test = X_test_fea;
            else:
                X_train = np.hstack((X_train,X_train_fea))
                X_test = np.hstack((X_test,X_test_fea))

        y_train = np.load('./TUH_y_train.npy')
        y_test = np.load('./TUH_y_test.npy') 
        
        return X_train, X_test, y_train, y_test 

def fit_by_label_gs(label, feature, samp, hypers=None):
    if label == 'TUH':
        X_train, X_test, y_train, y_test = generate_fit_data_tuh(label, feature, samp, cv=True)
    else:
        X_train, X_test, y_train, y_test = generate_fit_data(label, feature, samp, cv=True)
    
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
    print(gs.best_score_)
    print(gs.best_estimator_) 
    return gs 

def fit_by_label(label, feature, samp, params):
    X_train, X_test, y_train, y_test = generate_fit_data(label, feature, samp, r=1)
    gamma, C, kernel = params['gamma'], params['C'], params['kernel']
    svm = SVC(gamma=gamma, C=C, kernel=kernel, probability=True)
    svm.fit(X_train, y_train)
    return svm 

def predict_by_label(label, features, clf):
    test_data = get_data_by_label_type(label, 'test', features)
#    test_data = np.array(test_data)[:, 1:].astype(float)
    num = int((600 - 20) / 10 + 1)
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

#%%
la1 = 'Dog_5';
fea1 = ['band_power','freqcorr','hfd','hurst','pfd','IncAccEnergy'] #
#np.random.seed(42)
X_train, X_test, y_train, y_test = generate_fit_data(la1, fea1, 1, r=2, cv=True)
#X_train, X_test, y_train, y_test = generate_fit_data_tuh(la1, fea1, 1, cv=True)
#
from sklearn import preprocessing 
sca = True 
scaler = preprocessing.StandardScaler().fit(X_train)
(X_train, X_test) = (scaler.transform(X_train), scaler.transform(
        X_test)) if sca is True else (X_train, X_test)
 
#%%
# no selection 

clf1_lr = LogisticRegressionCV(penalty='l2', scoring='roc_auc')
clf1_svc = SVC(kernel='rbf', gamma=1e-3, C=3.0, probability=True) 
clf1_dt = RandomForestClassifier()

clf1 = clf1_lr
clf1.fit(X_train, y_train) 

r = clf1.predict(X_test) 
r1 = clf1.predict_proba(X_test)[:, 1]
print(confusion_matrix(r, y_test), roc_auc_score(y_test, r1))

#%% save the results of no selection
fpr, tpr, thresholds = roc_curve(y_test, r1) 
auc = roc_auc_score(y_test, r1) 
score = clf1.score(X_test, y_test) 

np.save('./tuh_processedfeature/tuh_parameter_results/rf_fpr_no_selection.npy',fpr)
np.save('./tuh_processedfeature/tuh_parameter_results/rf_tpr_no_selection.npy',tpr)
np.save('./tuh_processedfeature/tuh_parameter_results/rf_auc_no_selection.npy',auc)

#%% 
print(X_train.shape)
clf_lr = LogisticRegression(C=1.0, penalty='l2')
clf_svm = SVC(kernel='rbf', gamma=1e-3, C=3.0)
clf_dt = RandomForestClassifier() 

clf = clf_lr #choose one clf  
fsga = FeatureSelectionGA(
    clf, X_train, y_train 
) 
res = fsga.generate(20, cxpb = 0.5, mutxpb = 0.2, ngen=10)
res = np.array(res)
#%%
np.save('./processedfeature_Dog_3/parameter_results_ga/svm_selection_res_Dog_3.npy', res)

#%%
 
#with selection 

X_train_sel = X_train[:, res[0].astype(bool)] 
X_test_sel = X_test[:, res[0].astype(bool)]

X_train_sel_sca = X_train_sel 
X_test_sel_sca = X_test_sel 

clf1 = SVC(kernel='rbf', gamma=1e-3, C=3.0, probability=True)
clf2 = LogisticRegressionCV(penalty='l2', scoring='roc_auc') 
clf3 = RandomForestClassifier()
clf = clf2
print(clf) 
clf.fit(X_train_sel_sca, y_train) 

r = clf.predict(X_test_sel_sca)
r1 = clf.predict_proba(X_test_sel_sca)[:, 1]
print(confusion_matrix(r, y_test), roc_auc_score(y_test, r1)) 

#%%
import seaborn as sns
bands = ['CH2', 'CH8','CH16' ]
f = 'pfd'

bp = pd.read_csv(r'./processedfeature_Dog_3/Dog_3_{}_interictal.csv'.format(f, f))
bp = bp[bands] 
bp = bp[:1000] 
l = None 
labels = [] 
for b in bands:
    l = np.array(bp[b]) if l is None else np.hstack([l, np.array(bp[b])])
    labels += [b] * 1000
t = ['interictal'] * len(bands) * 1000

bp1 = pd.read_csv(r'./processedfeature_Dog_3/Dog_3_{}_preictal.csv'.format(f, f))
bp1 = bp1[bands] 
bp1 = bp1[:1000] 
l1 = None 
labels1 = [] 
for b in bands:
    l1 = np.array(bp1[b]) if l1 is None else np.hstack([l1, np.array(bp1[b])])
    labels1 += [b] * 1000
t1 = ['preictal'] * len(bands) * 1000

T = np.array(t + t1) 

LA = np.array(labels + labels1)  
L = np.hstack([l, l1])  
print(T.shape, LA.shape, L.shape)  
a = np.vstack([LA, T, L]).T 

X = pd.DataFrame(a, columns=['band', 'type', 'value']).convert_objects(convert_numeric=True)
ax = sns.violinplot(x="band", y="value", hue='type',
    data=X,  split=True, palette="Set2",
    scale="count", inner="quartile")

#%% corrletation matrix
#la1 = 'Dog_3';
#fea1 = ['band_power','IncAccEnergy','moment_2nd','moment_3rd','moment_4th',
#        'decorr','hfd','freqcorr','hurst','pfd'] #
#np.random.seed(42)
#X_train, X_test, y_train, y_test = generate_fit_data(la1, fea1, 1, r=2, cv=True)
#df = pd.DataFrame(moment_2nd_interictal, columns=feature_names)
import matplotlib.pyplot as plt
df = pd.DataFrame(X_train)

corr = df.corr() 

plt.figure()
plt.matshow(df.corr()) 
plt.colorbar()
plt.show()
 