#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:18:22 2019

@author: melizalab
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import os
import seaborn as sns
import toelis as tl
from scipy.io.wavfile import read

def spg(stim,Fs,downsample=True,resolution=50): 
    L=len(stim)
    dt = 1.0/Fs
    t=np.arange(0,L*dt,dt)
    nwindow = 256
    NFFT = 256
    nres = int(Fs/1000)
    noverlap = int(nwindow-nres)
    nbin = int(np.max(t)*1000)+1
    Ln = nbin*nres+noverlap
    Lx=int(Ln-L)
    signal = np.concatenate((stim,np.zeros(Lx)))
    Pxx, freqs, bins, im = plt.specgram(signal,NFFT=NFFT,Fs=Fs,noverlap=noverlap)   
    #cut off spectrogram at 8kHz
    Pxx = Pxx[:65,:]
    freqs = freqs[:65]
    Pxx = np.log10(Pxx)
    #squish spectrogram frequencies into the resolution of the RF
    if downsample == True:
        [Ls,Ts]=Pxx.shape
        nff=resolution
        df=freqs[-1]/nff
        f0=freqs[0]+df/2.
        lPxi=np.zeros((nff,Ts));
        fi=np.arange(f0,freqs[-1],df)
        for i in range(0,Ts):
            lPxi[:,i]=np.interp(fi,freqs,Pxx[:,i])
        Pxx=lPxi
        plt.cla()
    return (Pxx)

def auditory_plot(fp):
    fp = os.path.normpath(fp)
    files = glob.glob(fp+'/*.toe_lis')
    info = [os.path.basename(file) for file in files]
    info = [os.path.splitext(i)[0] for i in info]
    info = [i.split('_') for i in info]
    info = np.asarray(info)
    songs = np.unique(info[:,0])
    conditions = np.unique(info[:,1])
    for song in songs:
        fig, axes = plt.subplots(len(conditions)+1,1,sharex='all',figsize=(5,10))
        
        Fs,stim = read(os.path.join('../Recordings/Songs/'+song+'_gapnoise.wav'))
        Pxx = spg(stim,Fs)
        cmap = sns.cubehelix_palette(dark=0,light=1,as_cmap=True)
        xmin = 0.0
        xmax = Pxx.shape[1]/1000
        freqs = np.arange(0,10000,10000/Pxx.shape[0])
        extent = xmin,xmax,freqs[0],freqs[-1]
        imgplot = axes[0].imshow(np.flipud(10.*Pxx),aspect='auto',cmap=cmap,extent=extent,vmin=np.min(Pxx),vmax=np.max(Pxx))
        imgplot.set_clim(-10.,40.)
        axes[0].set_title(song)
        axes[0].get_xaxis().set_tick_params(direction='out')
        axes[0].get_yaxis().set_tick_params(direction='out')
        
        for i in range(len(conditions)):
            with open(os.path.join(fp,'_'.join((song,conditions[i]))+'.toe_lis')) as fs:
                times = tl.read(fs)
            raster = [time for sublist in times[0] for time in sublist]
            raster = np.asarray(raster)/1000.0
            profile = [len(times[0][x]) for x in range(len(times[0]))]
            y = np.repeat(np.arange(1,11),profile)
            axes[i+1].plot(raster,y,'|',color="black")
            axes[i+1].set_xlim([xmin,xmax])
            axes[i+1].set_ylim([0,10])
            axes[i+1].set_title(conditions[i])
            axes[i+1].get_xaxis().set_tick_params(direction='out')
            axes[i+1].get_yaxis().set_tick_params(direction='out')
        #fig.tight_layout()
        plt.savefig(os.path.join(fp,song+'.pdf'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory',help='File directory',required=True)
    args = parser.parse_args()
    auditory_plot(args.directory)