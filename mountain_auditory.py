#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:18:22 2019

@author: melizalab
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import glob
import os
import seaborn as sns
import toelis as tl
import json
from scipy.io.wavfile import read
from mountainlab_pytools import mdaio

def postprocess(bp):
    recordings = os.listdir(bp)
    for i in recordings:
        fp = os.path.join(bp,i)
        make_toes(fp)

def make_toes(fp):
    d = mdaio.readmda(os.path.join(fp,'mountainout/firings.curated.mda'))
    alignfile = glob.glob(os.path.join(fp,'*.align'))
    align = pd.read_csv(alignfile[0])
    clusters = np.unique(d[2])
    Fs = 30000
    for cluster in clusters:
        spikes = d[1,d[2]==cluster]
        channel = channel_map(d[0,d[2]==cluster][0])
        starts = np.asarray(align.total_start, dtype = int)
        #starts = np.asarray(align.total_pulse, dtype = int)
        stops = np.asarray(align.total_stop, dtype = int)
        #stops = np.asarray(align.total_pulse+(1.5*Fs), dtype = int)
        offsets = np.asarray(align.total_pulse, dtype = int)
        indices = [(spikes >= start) & (spikes <= stop) for start,stop in zip(starts,stops)]
        spiketrains = [spikes[ind] - offset for ind,offset in zip(indices,offsets)]
        dirname = os.path.join(fp, "ch%d_c%d" % (channel, cluster))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for stim in align['stim'].unique():
                f = [spiketrains[rec] for rec in align[align['stim']==stim].rec]
                #put together an informative  name for the file
                song = list(align.song[align.stim==stim].unique())[0]
                cond = list(align.condition[align.stim==stim].unique())[0]
                name = "%s_%s" % (song, cond)
                toefile = os.path.join(dirname, "%s.toe_lis" % name)
                with open(toefile, "wt") as ftl:
                    tl.write(ftl, np.asarray(f)/Fs*1000)
#        for song in align['song'].unique():
#            f = [spiketrains[rec] for rec in align[align['song']==song][align['condition']=='no-scene'].rec]
#            name = song+'_no-scene'
#            toefile = os.path.join(dirname, "%s.toe_lis" % name)
#            with open(toefile, "wt") as ftl:
#                tl.write(ftl, np.asarray(f)/Fs*1000)
#            f = [spiketrains[rec] for rec in align[align['song']==song][align['condition'].str.contains('scene63')].rec]
#            name = song+'_scene63'
#            toefile = os.path.join(dirname, "%s.toe_lis" % name)
#            with open(toefile, "wt") as ftl:
#                tl.write(ftl, np.asarray(f)/Fs*1000)
                
        auditory_plot(dirname)
        cluster_info(dirname,fp,cluster)
        
def cluster_info(dirname,fp,cluster):
    with open(os.path.join(fp,'mountainout/cluster_metrics.json')) as jsonfile:
        metrics = json.load(jsonfile)
    ci = metrics['clusters'][int(cluster)-1]
    json.dump(ci,open(os.path.join(dirname,'cluster_info.json'),"w"))
                    
def channel_map(channel):
    chmap = pd.read_csv('/home/melizalab/Data/probe-map.csv')
    probechannel = chmap[chmap.number==(channel-1)].probe
    return probechannel.values[0]

def spg(stim,Fs,downsample=False,resolution=50): 
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
    return (Pxx,freqs)

def auditory_plot(fp):
    fp = os.path.normpath(fp)
    files = glob.glob(fp+'/*.toe_lis')
    info = [os.path.basename(file) for file in files]
    info = [os.path.splitext(i)[0] for i in info]
    info = [i.split('_',1) for i in info]
    info = np.asarray(info)
    songs = np.unique(info[:,0])
    conditions = np.unique(info[:,1])
    #conditions = ['no-scene','scene63']
    for song in songs:
        fig, axes = plt.subplots(len(conditions)+1,1,sharex='all',figsize=(5,7))
        
        Fs,stim = read('../Recordings/Songs/'+song+'_gapnoise1.wav')
        #Fs,stim = read('../Chorus/'+song+'.wav')
        #Fs,scene = read('../Chorus/scene63_0.wav')
        #stimstart = (len(scene)-len(stim))//2
        #stimscene = np.zeros(len(scene))
        #stimscene[stimstart:stimstart+len(stim)] = stimscene[stimstart:stimstart+len(stim)]+stim
        #Pxx = spg(stimscene,Fs)
        Pxx,freqs = spg(stim,Fs)
        cmap = sns.cubehelix_palette(dark=0,light=1,as_cmap=True)
        xmin = 0.0
        xmax = Pxx.shape[1]/1000
        #freqs = np.arange(0,10000,10000/Pxx.shape[0])
        extent = xmin,xmax,freqs[0],freqs[-1]
        imgplot = axes[0].imshow(np.flipud(10.*Pxx),aspect='auto',cmap=cmap,extent=extent,vmin=np.min(Pxx),vmax=np.max(Pxx))
        imgplot.set_clim(-10.,40.)
        axes[0].set_title(song)
        axes[0].get_xaxis().set_tick_params(direction='out')
        axes[0].get_yaxis().set_tick_params(direction='out')
        
#        stimscene = stimscene+scene
#        Pxx = spg(stimscene,Fs)
#        xmax = Pxx.shape[1]/1000
#        freqs = np.arange(0,10000,10000/Pxx.shape[0])
#        extent = xmin,xmax,freqs[0],freqs[-1]
#        imgplot = axes[2].imshow(np.flipud(10.*Pxx),aspect='auto',cmap=cmap,extent=extent,vmin=np.min(Pxx),vmax=np.max(Pxx))
#        imgplot.set_clim(-10.,40.)
#        axes[2].set_title(song)
#        axes[2].get_xaxis().set_tick_params(direction='out')
#        axes[2].get_yaxis().set_tick_params(direction='out')
        
        for i in range(1,len(conditions)+1):
            with open(os.path.join(fp,'_'.join((song,conditions[i-1]))+'.toe_lis')) as fs:
                times = tl.read(fs)
            raster = [time for sublist in times[0] for time in sublist]
            raster = np.asarray(raster)/1000.0
            profile = [len(times[0][x]) for x in range(len(times[0]))]
            y = np.repeat(np.arange(1,11),profile)
            axes[i].plot(raster,y,'|',color="black",markersize=4)
            axes[i].set_xlim([xmin,xmax])
            axes[i].set_ylim([0,10])
            axes[i].set_title(conditions[i-1])
            axes[i].get_xaxis().set_tick_params(direction='out')
            axes[i].get_yaxis().set_tick_params(direction='out')
        #fig.tight_layout()
        plt.savefig(os.path.join(fp,song+'.pdf'))
        plt.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory',help='Base directory',required=True)
    args = parser.parse_args()
    postprocess(args.directory)