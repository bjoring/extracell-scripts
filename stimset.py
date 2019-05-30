#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:07:45 2018

@author: melizalab
"""
import numpy as np

def pulsestim(stim,unclip = False):
    if unclip == True:
        clips = np.where(stim > 2**15-1)
        for clip in clips:
            stim[clip] = 2**15-1
    pulse = np.zeros(len(stim))
    pulse[0] = 0.75
    if max(stim) > 2:
        pstim = np.transpose(np.vstack((np.asarray(stim)/(2**15-1),pulse)))
    else:
        pstim = np.transpose(np.vstack((np.asarray(stim),pulse)))
    return pstim

def gap(seed,f='../Stims1'):
    import glob
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
    
def rms(y):
    return (np.sqrt(np.mean(y.astype(float)**2)))

def dB(y):
    a0 = 0.00001*(2**15-1)
    return (20*np.log10(rms(y)/a0))

def scale(dB):
    a0 = 0.00001*(2**15-1)
    return (a0*(10**(dB/20)))
    
def dBclean(f='../Stims1'):
    import glob
    from scipy.io.wavfile import read
    
    songfiles = glob.glob(f+'/*.wav')
    songnames = [x.strip('stim.wav').split('/')[-1] for x in songfiles]
    songs = []
    for i in songfiles:
        fs,s = read(i)
        songs.append((s/rms(s))*scale(75))
    stims = np.asarray(songs)
    #stims = np.zeros((len(songs),len(songs[0]),2))
    #for i in range(len(songs)):
        #stims[i] = pulsestim(songs[i])
    return(stims,fs,songnames)
    
    
    