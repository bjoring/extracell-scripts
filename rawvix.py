#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:30:35 2018

@author: melizalab
"""

import numpy
import h5py
import matplotlib.pyplot as plt

with h5py.File('../O87_2018-04-12_15-26-05_clean/experiment1_prt0.arf', 'r') as f5:
    r29 = f5['rec_5']

    fig,axes = plt.subplots(32,sharex=True,sharey=True)
    for i in range(32):
        axes[i].plot(r29['channel'+str(i)][20000:60000])
plt.show()