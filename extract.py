# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:26:55 2018

@author: Jianhan Song
"""
from scipy import io, signal
import numpy as np
import pandas as pd

from Feature_Extractor import*
from SVC_Solver import*
from utility_functions import*
import constant


feature_names = ['IncAccEnergy']
## processing
for label in LABELS[3:4]:
    process_features_to_csv(
        feature_names, constant.DATA_FILE_PATH + '/' + label,
        label, type_list=None, dest_folder=constant.FEATURE_DEST_PATH)

# =============================================================================
# for label in LABELS[6:]:
#     process_features_to_csv(
#         feature_names, constant.DATA_FILE_PATH, label, type_list=['preictal'], dest_folder=constant.FEATURE_DEST_PATH)
# =============================================================================








