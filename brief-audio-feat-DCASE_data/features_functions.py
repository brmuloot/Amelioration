# -*- coding: utf-8 -*-
"""
Features functions and definitions

Created on Mon Oct 24 18:25:32 2022

@author: ValBaron10
"""

import numpy as np
from math import inf

def compute_features(sig_t, sig_s, sig_c, fs=51200):
    
    feature_functions = [min_sig,
                         max_sig,
                         mean_sig,
                         min_mean,
                         max_mean,
                         centroid,
                         RMS_sig,
                         std_sig,
                         mean_skewness,
                         mean_kurtosis,
                         skewness,
                         kurtosis,
                         shannon,
                         renyi,
                         rate_attack,
                         rate_decay,
                         silence_ratio,
                         threshold_crossing_rate
                         ]
    
    t = np.linspace(0, (sig_t.shape[0] - 1)*(1/fs), sig_t.shape[0])
    feat_temp = [(0, ()), (1, ()), (8, (t,)), (9, (t,)), (10, ()), (11, ()),
                 (12, (5,)), (12, (30,)), (12, (500,)), (13, (5,2)), (13, (30,2)),
                 (13, (500,2)), (13, (5,inf)), (13, (30,inf)), (13, (500,inf)),
                 (14, ()), (15, ()), (17, (0.25,)), (17, (0.5,)), (17, (0.75,))]
    
    freqs = np.linspace(0, fs/2, sig_s.shape[0])
    feat_spec = [(1, ()), (2, ()), (4, ()), (5, (freqs,)), (6, (freqs,)), (7, ()), (8, (freqs,)),
                 (9, (freqs,)), (10, ()), (11, ()), (12, (5,)), (12, (30,)), (12, (500,)),
                 (13, (5,2)), (13, (30,2)), (13, (500,2)), (13, (5,inf)),
                 (13, (30,inf)), (13, (500,inf)), (16, (0.05,)), (16, (0.1,)),
                 (16, (0.2,)), (17, (0.05,)), (17, (0.1,)), (17, (0.2,))]
    
    t_c = np.linspace(0, (sig_c.shape[0] - 1)*(1/fs), sig_c.shape[0])
    feat_ceps = [(1, ()), (2, ()), (4, ()), (5, (t_c,)), (6, (t_c,)), (7, ()), (8, (t_c,)),
                 (9, (t_c,)), (10, ()), (11, ()), (12, (5,)), (12, (30,)), (12, (500,)),
                 (13, (5,2)), (13, (30,2)), (13, (500,2)), (13, (5,inf)),
                 (13, (30,inf)), (13, (500,inf)), (15, ()), (16, (0.025,)),
                 (16, (0.05,)), (16, (0.075,)), (17, (0.025,)), (17, (0.05,)), (17, (0.075,))]
    
    N_feat = len(feat_temp) + len(feat_spec) + len(feat_ceps)
    
    # Compute the features
    features = []
    for feat_t in feat_temp:
        features.append(feature_functions[feat_t[0]](sig_t, feat_t[1]))
        
    for feat_s in feat_spec:
        features.append(feature_functions[feat_s[0]](sig_s, feat_s[1]))
        
    for feat_c in feat_ceps:
        features.append(feature_functions[feat_c[0]](sig_c, feat_c[1]))
    
    return N_feat, features
    


# Features functions    
def min_sig(signal, args):
    return np.min(signal)

def max_sig(signal, args):
    return np.max(signal)

def mean_sig(signal, args):
    return np.mean(signal)

def min_mean(signal, args):
    return np.min(signal) / np.mean(signal)

def max_mean(signal, args):
    return np.max(signal) / np.mean(signal)

def centroid(signal, args):
    sig_sq = signal**2
    E = sig_sq.sum()
    
    return 1/E*(args[0]*sig_sq).sum()

def RMS_sig(signal, args):
    sig_sq = signal**2
    E = sig_sq.sum()
    cent = centroid(signal, args)
    
    return np.sqrt(1/E*((args[0]**2)*sig_sq).sum() - cent**2)

def std_sig(signal, args):
    return np.std(signal)

def mean_skewness(signal, args):
    sig_sq = signal**2
    E = sig_sq.sum()
    cent = centroid(signal, args)
    B = RMS_sig(signal, args)
    sk_sq = ((args[0] - cent)**3 * sig_sq).sum() / (E*B**3)
    
    return np.sign(sk_sq)*np.sqrt(np.abs(sk_sq))

def mean_kurtosis(signal, args):
    sig_sq = signal**2
    E = sig_sq.sum()
    cent = centroid(signal, args)
    B = RMS_sig(signal, args)
    
    return np.sqrt(((args[0] - cent)**4 * sig_sq).sum() / (E*B**4))

def skewness(signal, args):
    mu = np.mean(signal)
    sigma = np.std(signal)
    
    return 1/signal.shape[0] * (((signal - mu)/sigma)**3).sum()

def kurtosis(signal, args):
    mu = np.mean(signal)
    sigma = np.std(signal)
    
    return 1/signal.shape[0] * (((signal - mu)/sigma)**4).sum()

def shannon(signal, args):
    counts, bins = np.histogram(signal, args[0])
    probas = counts / signal.shape[0]
    probas = probas[np.nonzero(probas)]
    
    return - (probas*np.log2(probas)).sum()

def renyi(signal, args):
    counts, bins = np.histogram(signal, args[0])
    probas = counts / signal.shape[0]
    probas = probas[np.nonzero(probas)]
    
    if args[1] == 2:
        return 1/(1-args[1]) * np.log2((probas**(args[1])).sum())
    
    elif args[1] == inf:
        return -np.log2(np.max(probas))

def rate_attack(signal, args):
    '''
    Be careful: I implemented the rate of decay as it is in AAA, but this
    implementation is not the same than the one defined in M. Malfante PhD
    The difference is that the signal is squared
    '''
    E_max = np.max(signal**2)
    diff = (signal**2)[1:] - (signal**2)[:-1]
    
    return np.max(diff) / E_max

def rate_decay(signal, args):
    '''
    Be careful: I implemented the rate of decay as it is in AAA, but this
    implementation is not the same than the one defined in M. Malfante PhD
    The difference is that the signal is squared, and the lowest increasing
    slop is taken instead of the highest decreasing one
    '''
    E_max = np.max(signal**2)
    diff = (signal**2)[1:] - (signal**2)[:-1]
    
    return np.min(diff) / E_max

def silence_ratio(signal, args):
    thresh_sig = np.clip(signal / np.max(signal), args[0], None)
    indexes = np.where(thresh_sig == args[0])[0]
    
    return indexes.shape[0] / signal.shape[0]

def threshold_crossing_rate(signal, args):
    sig_ctrd = signal / np.max(signal) - args[0]
    indexes = np.where(np.diff(np.sign(sig_ctrd)))[0]
    
    return indexes.shape[0] / signal.shape[0]
    