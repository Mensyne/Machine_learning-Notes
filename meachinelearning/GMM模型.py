#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:33:43 2020

@author: mensyne
"""

import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture as GMM


from sklearn.datasets.samples_generator import make_blobs

X,y_ture = make_blobs(n_samples = 700,centers = 4,cluster_std = 0.5,
                      random_state =2019)

X = X[:,::-1]

gmm = GMM(n_components = 4).fit(X)

labels = gmm.predict(X)

plt.scatter(X[:,0],X[:,1],c = labels,s = 5,cmap='viridis')

