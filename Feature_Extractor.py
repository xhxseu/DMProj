#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal, io, stats
import matplotlib.pyplot as plt
from itertools import combinations

SPECTRAL_BANDS = np.array([(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 48)])

class Feature_Extractor():
    
    def __init__(self, processed_data):
        self.processed_data = processed_data
        self.num_channels = len(processed_data['channels'])
        
    def extract_band_power(self):
        """Compute power of each spectral band for all channels given a
        10-min time-series data. Return a vector feature_data which is
        flattened by a no_of_channels by no_of_spectral_bands matrix.
        """
        processed_data = self.processed_data
        data = processed_data['data']
        sampling_frequency = processed_data['sampling_frequency']
        res = np.zeros((self.num_channels, len(SPECTRAL_BANDS)))

        for i in range(self.num_channels):
            f, Pxx_den = signal.welch(data[i, :], sampling_frequency, nperseg=int(2**8))           
            for j, spec in enumerate(SPECTRAL_BANDS):
                ind = np.where(np.logical_and(f >= spec[0], f < spec[1]))
                start = np.interp(spec[0], f, Pxx_den)
                end = np.interp(spec[1], f, Pxx_den)
                x0 = np.insert(np.array([spec[0], spec[1]]), 1, f[ind])
                y0 = np.insert(np.array([start, end]), 1, Pxx_den[ind])
                res[i, j] = np.trapz(y0, x0)  # integration

        return res.flatten()

    def extract_stat_moments(self):
        data = self.processed_data['data']
        res = np.zeros((2, self.num_channels))
        for i in range(self.num_channels):
            channeldata = data[i, :]
            res[0, i] = stats.skew(channeldata)
            res[1, i] = stats.kurtosis(channeldata)
        return res.flatten()
    

    def band_power_feature_names(num_of_channels):
        """Return corresponding feature names of data computed
        by extract_band_power, e.g., CH1_band1.
        """
        feature_names = [] 
        for i in range(num_of_channels): 
            for j, spec in enumerate(SPECTRAL_BANDS): 
                feature_names.append('CH{}_band{}'.format(i + 1, j + 1)) 
        return feature_names 

    def stat_moments_feature_names(num_of_channels):
        feature_names = []
        for j, spec in enumerate(['Skewness', 'Kurtosis']):
            for i in range(num_of_channels):
                feature_names.append('CH{}_{}'.format(i + 1, spec))
        return feature_names
    
    def extract_IncAccEnergy(self):
        """Compute 'increments of the accumulated energy', see ref. Maiwald 2004.
        """
        processed_data = self.processed_data
        data = processed_data['data']
        sampling_frequency = processed_data['sampling_frequency']
        
        time_window_length = int(1.25 * sampling_frequency)
        time_window_shift = int(0.45 * sampling_frequency)
        n_window = int(data.shape[1] * 1.0 /time_window_shift) +1
        n_iAE = int(n_window / 10.0)
        iAE = np.zeros(data.shape[0])
        for ii in range(data.shape[0]):
            #  energy of window-k
            Ek = np.zeros(n_window)
            for k in range(n_window): 
                data_k = data[ii, (k * time_window_shift):min(k * time_window_shift + time_window_length, data.shape[1]-1)]
                Ek[k] = np.mean(np.square(data_k)) 
            iAE_temp = np.zeros(n_iAE)
            for m in range(n_iAE):
                iAE_temp[m] = np.mean(Ek[m * 10 : (m + 1) * 10 ]) 
            
            iAE[ii] = max(iAE_temp)

        return iAE.flatten()

    def extract_HMHC(self):
        
        processed_data = self.processed_data
        data = processed_data['data']
        sampling_frequency = processed_data['sampling_frequency']
        
        HMHC = np.zeros((data.shape[0],2))
        
#        for ii in range(data.shape[0]):
#            sk = np.fft.fft(data[ii,:])
#            N = int(data.shape[1]/2.0)
#            pk = np.absolute(sk[0:N])**2
#            P = np.sum(pk)
#            HMHC[ii,0] = np.sum(np.arange(N)**2 * pk) / P
#            HMHC[ii,1] = np.sum(np.arange(N)**4 * pk) / P
        
        for i in range(num_of_channels):
            f, Pxx_den = signal.welch(data[i, :], sampling_frequency, nperseg=int(2**8))           
            ind = np.where(np.logical_and(f >= 0, f < sampling_frequency/2.0))
            start = np.interp(0, f, Pxx_den)
            end = np.interp(sampling_frequency/2.0, f, Pxx_den)
            x0 = np.insert(np.array([0, sampling_frequency/2.0]), 1, f[ind])
            y0 = np.insert(np.array([start, end]), 1, Pxx_den[ind])
            P = np.trapz(y0, x0)  # integration
            
            HMHC[i,0] = np.sum(np.arange(len(y0))**2 * y0) / P
            HMHC[i,1] = np.sum(np.arange(len(y0))**4 * y0) / P
            
        return HMHC.flatten() 
    
    def IncAccEnergy_feature_names(num_of_channels):
        """Return corresponding feature names of data computed
        by extract_IncAccEnergy, e.g., CH1_iAE.
        """         
        feature_names = []
        for i in range(num_of_channels): 
            feature_names.append('CH{}_iAE'.format(i + 1))
        return feature_names
    
    def HMHC_feature_names():
        """Return corresponding feature names of data computed
        by extract_HMHC, e.g., CH1_iAE.
        """         
        feature_names = [] 
        for i in range(num_of_channels): 
            for j, feat in enumerate(['HM','HC']):
                feature_names.append('CH{}_'.format(i + 1)+feat)
        return feature_names
    
    
    
    
    
    
    
    
    

