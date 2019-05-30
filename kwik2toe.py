#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:48:46 2018

@author: melizalab
"""

import toelis as tl
import h5py
import numpy as np
import pandas as pd
import argparse
import os
import inspect_auditory as ia

Fs = 30000.0

def make_toes(fp,t,s):
    
    #either do all the shanks or just the one specified
    if s == 4:
        shanks = range(4)
    elif s == 8:
        shanks = range(8)
    else:
        shanks = [s]
        
    #get the base name of the file
    base = os.path.splitext(os.path.basename(fp))[0]
    
    #read in the csv with the clusters of interest and the csv with the correct alignments
    ref = pd.read_csv('../clusterref.csv')
    alignfile = os.path.join(os.path.dirname(fp), base) + ".align"
    align = pd.read_csv(alignfile)
    
    for shank in shanks:
        print(shank)
        #open the .kwik file and pull out the lists of spikes and clusters
        with h5py.File(fp) as kwik:
            spikes = np.asarray(kwik['channel_groups/'+str(shank)+'/spikes/time_samples'])
            clusters = np.asarray(kwik['channel_groups/'+str(shank)+'/spikes/clusters/main'])
        #get the list of clusters to analyze
        if t == 'all':
            units = list(ref.cluster[ref.recording == base][ref.shank==shank])
        else:
            units = list(ref.cluster[ref.recording == base][ref.shank==shank][ref.type==t])
        #for each cluster, write the toelis files to a folder    
        for clust in units:
            #get only the spikes that are part of one cluster
            spikeset = (spikes[clusters==clust])/Fs*1000.0
            #spikeset = [spikes for spikes, clusters in zip(spikes, clusters) if clusters==clust]
            #separate the spikes by recording number
            s = []
            # TODO: consider iterating through the rows to access all fields
            for rec in align['rec'].index:
                start = int(align[align.rec==rec].total_start)/Fs*1000.0
                stop =  int(align[align.rec==rec].total_stop)/Fs*1000.0
                offset = int(align[align.rec==rec].total_pulse)/Fs*1000.0
                ind = (spikeset >= start) & (spikeset <= stop)
                
                s.append(spikeset[ind] - offset)
                #s.append([spike-int(align[align.rec==rec].total_start) for spike in spikeset if spike >= int(align[align.rec==rec].total_start) and spike <= int(align[align.rec==rec].total_stop)])
            dirname = os.path.join(os.path.dirname(fp), "s%d_c%d" % (shank, clust))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            #group repeated trials of the same stimulus and write to file
            for stim in align['stim'].unique():
                f = []
                for rec in align[align['stim']==stim].rec:
                    f.append(np.asarray(s[rec]))
                #put together an informative  name for the file
                song = list(align.song[align.stim==stim].unique())[0]
                cond = list(align.condition[align.stim==stim].unique())[0]
                name = "%s_%s" % (song, cond)
                toefile = os.path.join(dirname, "%s.toe_lis" % name)
                with open(toefile, "wt") as ftl:
                    tl.write(ftl, np.asarray(f))
            ia.auditory_plot(dirname)
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',help='File name',required=True)
    parser.add_argument('-t','--type',help='Unit type',required=False,default='all')
    parser.add_argument('-s','--shank',help='Shank number (0-3)',required=False,type=int,default=4)
    args = parser.parse_args()
    make_toes(args.file,args.type,args.shank)