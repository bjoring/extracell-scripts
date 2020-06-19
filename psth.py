# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:36:51 2019

@author: Margot
"""

import numpy as np
import os
import sys
import glob
import pandas as pd
import json
import pyspike as spk
import re

sys.path.append("../induction/induction")
import core


def get_pprox(path):
    pproxx = glob.glob(os.path.join(path,'*.pprox'))
    pproxx.sort()
    rate_cont = []
    for px in pproxx:
        print(px)
        rate_cont.extend(rate_select(px))
    rate_info = pd.DataFrame.from_records(data = rate_cont, columns = ["Bird","Recording","Channel","Cluster","Song","Condition","AvgRate","SDRate","Selectivity","Lag","Start","Stop",'p_continuous','p_gap','p_noise',
                                                                             'd_cont_gap','d_cont_gap_norm',
                                                                             'd_cont_gn','d_cont_gn_norm',
                                                                             'd_gap_gn','d_gap_gn_norm',
                                                                             'd_noise_gn','d_noise_gn_norm',
                                                                             'd_cn_gn','d_cn_gn_norm'])
    return rate_info
            
            
def rate_select(path): 
    with open(path,'r') as fp:
        px = json.load(fp)
         
    #syll = pd.read_csv('../restoration_syllables.csv')
    
    song = []
    condition = []
    rate = []
    gapon = {}
    #gapoff = {}
    spikes = []
    
    for t in range(len(px['pprox'])):
        #For new recordings:
        if px['pprox'][t]['condition'] == 'continuous':
            song.append(px['pprox'][t]['stimulus']+'-1')
            song.append(px['pprox'][t]['stimulus']+'-2')
            condition.append(px['pprox'][t]['condition']+'1')
            condition.append(px['pprox'][t]['condition']+'2')
            spikes.append(px['pprox'][t]['event'])
            spikes.append(px['pprox'][t]['event'])
        else:
            songid = px['pprox'][t]['stimulus']+'-'+px['pprox'][t]['condition'][-1]
            song.append(px['pprox'][t]['stimulus']+'-'+px['pprox'][t]['condition'][-1])
            condition.append(px['pprox'][t]['condition'])
            spikes.append(px['pprox'][t]['event'])
            
        if 'gap_on' in px['pprox'][t].keys():
            gapon[songid] = px['pprox'][t]['gap_on']
            #gapoff[songid] = px['pprox'][t]['gap_off']
            
    x = []
    #y = []
    for s in song:
        x.append(gapon[s])
        #y.append(gapoff[s])
        
    gapon = x
    #gapoff = y
    lag = []
    train = []
    gaps = difference_psth(song, condition, spikes, gapon)
    songset = np.unique(song)
    for t in range(len(spikes)):        
        windowstart = gaps[np.where(songset == song[t])[0][0]][0]
        windowstop = gaps[np.where(songset == song[t])[0][0]][1]
        interval = (windowstop/1000) - (windowstart/1000)
        if windowstart != 0:
            numspikes = len([spike for spike in spikes[t] if spike >= windowstart and spike <= windowstop])
            rate.append(numspikes/interval)
            lag.append(windowstart-gapon[t])
            train.append(spk.SpikeTrain([spike for spike in spikes[t] if spike >= windowstart and spike <= windowstop],[windowstart,windowstop]))
        else:
            rate.append(np.nan)
            lag.append(np.nan)
            train.append(np.nan)
        
    ziplist = list(zip(song, 
                       condition, 
                       rate,
                       lag))
        
    df = pd.DataFrame(ziplist, columns = ["Song","Condition","Rate","Lag"])
    avgrate = df.groupby(['Song','Condition']).mean()
    avgrate = [(avgrate.iloc[x].name + (avgrate.iloc[x].values[0],)) for x in range(len(avgrate))]
    sdrate = df.groupby(['Song','Condition']).std().values
    lag = df.groupby(['Song','Condition']).mean().values
    ri = df[(df['Condition']=='continuous1') | (df['Condition']=='continuous2')].groupby(['Song','Condition']).mean()
    count = 16
    selectivity = (1-((np.sum(ri/count))**2)/(np.sum((ri**2)/count)))/(1-(1/count))
    selectivity = [selectivity]*len(avgrate)
    
    bird = [px['bird']]*len(avgrate)
    recording = [px['recording']]*len(avgrate)
    channel = [px['channel']]*len(avgrate)
    cluster = [px['cluster']]*len(avgrate)
    k = 5
    window = [gap for gap in gaps for i in range(k)]
    
    confusion = spikey(train,song,condition)
    confusion = [dist for dist in confusion for i in range(k)]
    
    records = list(zip(bird,
                    recording,
                    channel,
                    cluster,
                    avgrate,
                    sdrate,
                    selectivity,
                    lag,
                    window,
                    confusion))
    
    return [(a,b,c,d,e,f,g,h[0],i.values[0],j[1],k,l,m,n,o,p,q,r,s,t,u,v,w,x,y) for a,b,c,d,(e,f,g),h,i,j,(k,l),[m,n,o,p,q,r,s,t,u,v,w,x,y] in records]
    
def difference_psth(song, condition, spikes, gapon):
    
    songset = np.unique(song)
    psths = []
    binwidth = 100
    overlap = 90
    window = []
    
    for s in songset:
        continuous = []
        gap = []
        for i in range(len(spikes)):
            if song[i] == s and (condition[i] == 'continuous1' or condition[i] == 'continuous2'):
                continuous.extend(spikes[i])
            elif song[i] == s and (condition[i] == 'gap1' or condition[i] == 'gap2'):
                gapguess = gapon[i]
                gap.extend(spikes[i])
        continuous.sort()
        gap.sort()
        if not continuous and not gap:
            window.append((0,0))
            continue
        elif not continuous or not gap:
            if not continuous:
                spikelen = int(np.max(gap))
            else:
                spikelen = int(np.max(continuous))
        elif np.max(continuous) > np.max(gap):
            spikelen = int(np.max(continuous))
        else:
            spikelen = int(np.max(gap))
        cpsth = window_psth(continuous, spikelen, binwidth, overlap)
        gpsth = window_psth(gap, spikelen, binwidth, overlap)
        diff_psth = np.abs(cpsth - gpsth)
        psths.append(diff_psth)
        baseline = np.median(diff_psth)*2
        high = np.median(diff_psth)*5
        #baseline = np.std(diff_psth)
        #high = baseline*3
        peaks = np.where(diff_psth > high)
        peaks = peaks[0]
        if len(peaks) > 0:
            start = [p for p in peaks if p*(binwidth-overlap) >= gapguess and p*(binwidth-overlap) <= np.max(gapon)+100]
            if len(start) > 0:
                start = start[0]
                stop = window_end(diff_psth, start, baseline)
                window.append((start*(binwidth-overlap),stop*(binwidth-overlap)))
            else: window.append((0,0))
        else:
            window.append((0,0))
    return window

def window_end(psth, start, baseline):
    count = 0
    index = 0
    for i in np.arange(start,len(psth)):
        if index == 0 and psth[i] < baseline:
            count += 1
            index = i
        elif index == i-1 and psth[i] < baseline:
            count += 1
            index = i
        else:
            count = 0
            index = 0
        if count == 3:
            return index
    return len(psth)-1
            

def window_psth(spikes, length, binwidth=100, overlap = 75):
    spikerate = []
    for b in np.arange(0, length-binwidth, binwidth-overlap):
        spikerate.append(len([spike for spike in spikes if spike >= b and spike < b+binwidth]))
    return(np.asarray(spikerate))
    
def spikey(train,song,condition):  
    songset = np.unique(song)
    dist = []
    for s,stim in enumerate(songset):
        subset = np.where(np.asarray(song) == stim)[0]
        trainsub = [train[x] for x in subset if train[x] is not np.nan]
        labels = [condition[x] for x in subset if train[x] is not np.nan]
        pairs = np.zeros((len(trainsub),len(trainsub)))
        for i in range(len(trainsub)):
            for j in range(len(trainsub)):
                pairs[i,j] = spk.spike_distance(trainsub[i], trainsub[j])
        df = pd.DataFrame(pairs, columns = labels, index = labels)
        if trainsub:
            dist.append(compute_dist(df))
        else:
            dist.append([0,0,0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
    return dist
        #df.to_csv(os.path.splitext(eventfile)[0]+'_'+stim+'.csv')
    
def compute_dist(dist):

    # repair column names
    #dist.columns = pd.Index([x.split(".")[0] for x in dist.columns])
    dist.columns = pd.Index([re.findall(r'(\w+?)(\d+)', x)[0][0] for x in dist.columns])
    dist.index = [re.findall(r'(\w+?)(\d+)', x)[0][0] for x in dist.index]
    # calculate trial-cluster distances
    tcdist = core.trial_cluster_distance(dist)
    # calculate cluster-cluster distances
    ccdist = core.cluster_distance(tcdist)
    # probabilities of confusing gapnoise with continuous, gap, or noise
    gap_conf = core.confusion(tcdist, "gapnoise", ["continuous", "gap", "noise"])
    gap_conf = gap_conf / gap_conf.sum()
    # raw distance between gap and continuous
    gap_cont_dist = ccdist.loc['continuous', 'gap']
    # normalize by pooled intracluster distance
    gap_cont_pool= np.sqrt(ccdist.loc['continuous', 'continuous']**2 + ccdist.loc['gap', 'gap']**2)
    #raw distance between gap_noise and continuous
    gn_cont_dist = ccdist.loc['continuous','gapnoise']
    gn_gap_dist = ccdist.loc['gap','gapnoise']
    gn_noise_dist = ccdist.loc['noise','gapnoise']
    gn_cn_dist = ccdist.loc['continuousnoise','gapnoise']
    #normalized gap_noise and continuous distance
    gn_cont_pool = np.sqrt(ccdist.loc['continuous', 'continuous']**2 + ccdist.loc['gapnoise', 'gapnoise']**2)
    gn_gap_pool = np.sqrt(ccdist.loc['gap', 'gap']**2 + ccdist.loc['gapnoise', 'gapnoise']**2)
    gn_noise_pool = np.sqrt(ccdist.loc['noise', 'noise']**2 + ccdist.loc['gapnoise', 'gapnoise']**2)
    gn_cn_pool = np.sqrt(ccdist.loc['continuousnoise', 'continuousnoise']**2 + ccdist.loc['gapnoise', 'gapnoise']**2)
    # output: name, p_continuous, p_gap, p_noise, d_cont_gap, d_cont_gap_norm
    return gap_conf.tolist() + [gap_cont_dist, gap_cont_pool, 
                          gn_cont_dist, gn_cont_pool, 
                          gn_gap_dist, gn_gap_pool, 
                          gn_noise_dist, gn_noise_pool, 
                          gn_cn_dist, gn_cn_pool]
    

if __name__ == "__main__":
    
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pproxdir", help="directory of pprox files (JSON)")
    args = p.parse_args()
    args.distdir = os.path.normpath(args.pproxdir)
    try:
        rate_info = get_pprox(args.pproxdir)
        rate_info.to_csv(os.path.join(os.path.dirname(args.pproxdir), 'rate_info.csv'))
    except Exception as e:
        print("%s: %s" % (args.distdir, e), file=sys.stderr)
        
