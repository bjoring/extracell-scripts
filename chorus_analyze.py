#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:18:47 2019

@author: melizalab
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import toelis as tl
import os
import pandas as pd

units = glob.glob('../chorus_pilot/auditory/*ch*')
stims = ['B2','R56','R180','R253']
stimlen = 1500
df = pd.DataFrame(index=np.arange(0, len(units)), columns = ['unit','stim48','stim53','stim58','stim63','stim68','stim73','stim78','scene48','scene53','scene58','scene63','scene68','scene73','scene78','scene'])
for i,unit in enumerate(units):
    stimx = np.zeros((4,7))
    scenex = np.zeros((4,7))
    for j,stim in enumerate(stims):
        data = glob.glob(unit+'/'+stims[j]+'*toe_lis')
        data.sort()
        for k,d in enumerate(data):
            with open(d, 'r') as fp:
                spikes = tl.read(fp)
            rate = [np.count_nonzero(train<stimlen) for train in spikes[0]]
            rate = sum(rate)/(stimlen/1000)/10
            if 'no-scene' in d:
                stimx[j,k//2] = rate
            else:
                scenex[j,k//2] = rate
    np.savetxt(os.path.join(unit, 'stimrates.csv'), stimx, delimiter=',')
    np.savetxt(os.path.join(unit, 'scenerates.csv'), scenex, delimiter=',')
    
    stimavg = np.average(stimx, axis = 0)
    sceneavg = np.average(scenex, axis = 0)
    stimsceneavg = sceneavg - sceneavg[0]
    
    row = [os.path.basename(unit)]
    row.extend(stimavg)
    row.extend(stimsceneavg)
    row.append(sceneavg[0])
    
    df.loc[i] = row

df.to_csv('../chorus_pilot/auditory/summary.csv',index=False)