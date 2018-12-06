#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:07:45 2018

@author: melizalab
"""

def pulsestim(stim):
    import numpy as np
    pulse = np.zeros(len(stim))
    pulse[0] = 1
    pstim = np.transpose(np.vstack((np.asarray(stim)/(2**15-1),pulse)))
    return pstim

def gap(seed,f='../Stims1'):
    import glob
    import numpy as np
    import pandas as pd
    from scipy.io.wavfile import read
    
    syllables = pd.read_csv('../syllables.csv')
    songfiles = glob.glob(f+'/*.wav')
    songnames = [x.strip('stim.wav').split('/')[-1] for x in songfiles]
    songs = []
    for i in songfiles:
        fs,s = read(i)
        songs.append(s)
    stims = np.zeros((len(songs)*2,len(songs[0]),2))
    np.random.seed(seed)
    for i in range(len(songs)):
        block = np.random.choice(syllables.loc[syllables.songid==songnames[i]].index,1)
        #print(syllables.loc[gap])
        start = int(syllables.get_value(int(block),'start')*fs)
        stop = int(syllables.get_value(int(block),'end')*fs)
        if stop-start > 4000:
            stop = start+4000
        g = syllables.loc[block]
        g['gstart'] = start
        g['gstop'] = stop
        if i == 0:
            g.to_csv('../gaps.csv',sep=',',index=False,mode='w')
        else:
            g.to_csv('../gaps.csv',sep=',',index=False,header=False,mode='a')
        wn = np.random.normal(0,2**15-1,size=stop-start)
        wn = wn/np.max(np.abs(wn))*(2**15-1)*0.8
        gstim = np.concatenate((songs[i][:start],np.zeros(stop-start),songs[i][stop:]))
        nstim = np.concatenate((songs[i][:start],wn,songs[i][stop:]))
        stims[i*2] = pulsestim(gstim)
        stims[i*2+1] = pulsestim(nstim)
    return(stims,fs,songnames)
    
def clean(f='../Stims1'):
    import glob
    import numpy as np
    from scipy.io.wavfile import read
    
    songfiles = glob.glob(f+'/*.wav')
    songnames = [x.strip('stim.wav').split('/')[-1] for x in songfiles]
    songs = []
    for i in songfiles:
        fs,s = read(i)
        songs.append(s)
    stims = np.asarray(songs)
    #stims = np.zeros((len(songs),len(songs[0]),2))
    #for i in range(len(songs)):
        #stims[i] = pulsestim(songs[i])
    return(stims,fs,songnames)
    
    
    