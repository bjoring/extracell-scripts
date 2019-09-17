#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:55:59 2019

@author: melizalab
"""

import pyspike as spk
import toelis as tl
import numpy as np
import os
import glob
import argparse
import pandas as pd
import json

def analyze(bp, uuid, age, typ, sex):
    recordings = os.listdir(bp)
    for i in recordings:
        fp = os.path.join(bp,i)
        clusters = glob.glob(os.path.join(fp,'ch*_c*'))
        clusters = [cluster for cluster in clusters if os.path.isdir(cluster)]
        for cluster in clusters:
            make_pprox(cluster,fp,uuid,age,typ,sex)
            spikey(fp)
            
def spikey(fp):
    events = glob.glob(os.path.join(fp,'*.pprox'))
    events.sort()
    #For new recordings:
    syll = pd.read_csv('../restoration_syllables.csv')
    
    #For old recordings:
    #syll = pd.read_csv('../syllables.csv')
    
    for eventfile in events:
        
        with open(eventfile) as fl:
            data = json.load(fl)
        song = []
        condition = []
        train = []
        gapon = {}
        gapoff = {}
        spikes = []
        for t in range(len(data['pprox'])):
            #For new recordings:
            if data['pprox'][t]['condition'] == 'continuous':
                song.append(data['pprox'][t]['stimulus']+'-1')
                song.append(data['pprox'][t]['stimulus']+'-2')
                condition.append(data['pprox'][t]['condition']+'1')
                condition.append(data['pprox'][t]['condition']+'2')
                spikes.append(data['pprox'][t]['event'])
                spikes.append(data['pprox'][t]['event'])
            else:
                songid = data['pprox'][t]['stimulus']+'-'+data['pprox'][t]['condition'][-1]
                song.append(data['pprox'][t]['stimulus']+'-'+data['pprox'][t]['condition'][-1])
                condition.append(data['pprox'][t]['condition'])
                spikes.append(data['pprox'][t]['event'])
                
            if 'gap_on' in data['pprox'][t].keys():
                gapon[songid] = data['pprox'][t]['gap_on']
                gapoff[songid] = data['pprox'][t]['gap_off']
                
            #For old recordings:
            #song.append(data['pprox'][t]['stimulus'])
            #condition.append(data['pprox'][t]['condition'])
            #spikes.append(data['pprox'][t]['event'])
            #if 'gap_on' in data['pprox'][t].keys():
                #gapon[song[t]] = data['pprox'][t]['gap_on'][0]/40
                #gapoff[song[t]] = data['pprox'][t]['gap_off'][0]/40
                
        songset = np.unique(song)
        x = []
        y = []
        for s in song:
            x.append(gapon[s])
            y.append(gapoff[s])
            
        gapon = x
        gapoff = y
        
        for t in range(len(spikes)):
            #For new recordings:
            syllstart = syll['start'][syll['songid'] == song[t][:-2]][syll['start'] <= gapon[t]/1000+0.001][syll['end'] >= gapoff[t]/1000-0.001].values[0] * 1000
            index = syll[syll['songid'] == song[t][:-2]][syll['start'] <= gapon[t]/1000+0.001][syll['end'] >= gapoff[t]/1000-0.001].index.values[0] + 1
            
            #For old recordings:
            #syllstart = syll['start'][syll['songid'] == song[t]][syll['start'] <= gapon[t]/1000+0.001][syll['end'] >= gapoff[t]/1000-0.001].values[0] * 1000
            #index = syll[syll['songid'] == song[t]][syll['start'] <= gapon[t]/1000+0.001][syll['end'] >= gapoff[t]/1000-0.001].index.values[0] + 1
            
            nextsyllend = syll['end'].at[index] * 1000
            spikes[t] = [spike for spike in spikes[t] if spike >= syllstart and spike <= nextsyllend]
            train.append(spk.SpikeTrain(spikes[t],[syllstart,nextsyllend]))
            
        for s,stim in enumerate(songset):
            pairs = np.zeros((len(train)//len(songset),len(train)//len(songset)))
            subset = np.where(np.asarray(song) == stim)
            trainsub = [train[x] for x in subset[0]]
            for i in range(len(trainsub)):
                for j in range(len(trainsub)):
                    pairs[i,j] = spk.spike_distance(trainsub[i], trainsub[j])
            labels = [condition[x] for x in range(len(condition)) if x in subset[0]]
            df = pd.DataFrame(pairs, columns = labels, index = labels)
            df.to_csv(os.path.splitext(eventfile)[0]+'_'+stim+'.csv')


        
def make_pprox(cluster,fp,uuid,age,typ,sex):
    _schema = "https://meliza.org/spec:2/pprox.json#"
    obj = {"$schema": _schema}
    #obj.update(args.metadata)
    
    logfile = glob.glob(os.path.join(fp,'*.log'))[0]
    with open(logfile, 'r') as fpj:
        log = json.load(fpj)
        
    alignfile = glob.glob(os.path.join(fp,'*.align'))[0]
    align = pd.read_csv(alignfile)
    
    stimfile = pd.read_csv('../stimuli.csv')
    
    obj['bird'] = log['bird']
    obj['bird-uuid'] = uuid
    obj['bird-age'] = age
    obj['bird-type'] = typ
    obj['bird-sex'] = sex
    experiment = log['stimtype']
    obj['experiment'] = experiment
    if experiment == 'induction':
        familiar = os.path.basename(log['stimset'])
        obj['familiar'] = familiar
    obj['songs'] = log['songs']
    obj['recording'] = os.path.basename(fp).split('_')[-1]
    obj['channel'] = os.path.basename(cluster).split('_')[0].strip('ch')
    obj['cluster'] = os.path.basename(cluster).split('_')[-1].strip('c')
    obj['seed'] = log['seed']
    obj['location'] = log['location']
    obj['hemisphere'] = log['hemisphere']
    obj['x-coord'] = log['x-coordinates']
    obj['y-coord'] = log['y-coordinates']
    obj['z-coord'] = log['z-coordinates']
    
    pprox = []
    
    for fname, eventlist in load_toelis(cluster):
        stim,condition = fname.split('_',1)
        for trial, events in enumerate(eventlist):
            pproc = {"trial": trial,
                     "stimulus": stim,
                     "units": "ms",
                     "event": events.tolist()
                    }
            if experiment == 'induction':
                old = True if familiar == 'Stims1' else False
                if len(np.unique(align['stim'][align['song'] == stim][align['condition'] == condition])) == 1:
                    stimnum = np.unique(align['stim'][align['song'] == stim][align['condition'] == condition])[0]
                else:
                    print("warning: %s is not mapped to a unique stim number in align file", fname)
                pproc["condition"] = condition
                pproc["category"] = 'familiar' if (stimfile['set'][stimfile['stimulus'] == fname] == familiar).any() else 'unfamiliar'
                pproc["stim_on"] = 0
                if old == True:
                    pproc["stim_off"] = stimfile['length'][stimfile['stimulus'] == fname][stimfile['set'] == 'pilot'].values[0]
                    pproc["stim_uuid"] = stimfile['uuid'][stimfile['stimulus'] == fname][stimfile['set'] == 'pilot'].values[0]
                    if log['presentation'][str(stimnum)].get('gaps'):
                        pproc["gap_on"] = [log['presentation'][str(stimnum)]['gaps'][0][0],log['presentation'][str(stimnum)]['gaps'][1][0]]
                        pproc["gap_off"] = [log['presentation'][str(stimnum)]['gaps'][0][1],log['presentation'][str(stimnum)]['gaps'][1][1]]
                else:
                    pproc["stim_off"] = stimfile['length'][stimfile['stimulus'] == fname][stimfile['set'] != 'pilot'].values[0]
                    pproc["stim_uuid"] = stimfile['uuid'][stimfile['stimulus'] == fname][stimfile['set'] != 'pilot'].values[0]
                    if log['presentation'][str(stimnum)].get('gaps'):
                        pproc["gap_on"] = log['presentation'][str(stimnum)]['gaps'][0]/40000*1000
                        pproc["gap_off"] = log['presentation'][str(stimnum)]['gaps'][1]/40000*1000
            elif experiment == 'chorus':
                pproc["condition"] = "no scene" if condition == 'no-scene' else "scene"
                pproc["stim_on"] = (1500 - stimfile['length'][stimfile['stimulus'] == stim].values[0]) // 2
                pproc["stim_off"] = pproc["stim_on"] + stimfile['length'][stimfile['stimulus'] == stim].values[0]
                pproc["stim_uuid"] = stimfile['uuid'][stimfile['stimulus'] == stim].values[0]
                if pproc["condition"] == "scene":
                    pproc["chorus_on"] = 0
                    pproc["chorus_off"] = 1500
                #pproc["chorus_uuid"] = stimfile['uuid'][stimfile['stimulus'] == condition]
                pproc["intensity"] = stim[-2:]
            else:
                print("warning: unknown experiment type %s" % experiment)
            pprox.append(pproc)
    
    obj["pprox"] = pprox
    #obj["pprox"] = list(load_toelis(cluster))
    
    outfile = os.path.join(fp,'_'.join((obj['bird'],obj['recording'],os.path.basename(cluster)))+'.pprox')
    with open(outfile, 'w') as fpx:
        json.dump(obj, indent=None, fp=fpx)
        
    print("Written to %s" % outfile)
        
def load_toelis(dirname):
    """ Returns an interator that will go through all the toelis files in dirname """
    for toefile in glob.iglob(os.path.join(dirname, "*.toe_lis")):
        fname = os.path.splitext(os.path.basename(toefile))[0]
        with open(toefile, "rt") as fpt:
            for unit, eventlist in enumerate(tl.read(fpt)):
                if unit > 0:
                    print("warning: %s toelis file has more than one unit" % toefile)
                yield fname, eventlist
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help = 'Base directory', required = True)
    parser.add_argument('-u', '--uuid', help = 'Bird UUID', required = False, default = 'NA')
    parser.add_argument('-a', '--age', help = 'Bird age (days post-hatch)', required = False, default = 'NA')
    parser.add_argument('-t', '--type', help = 'Bird type (CR, FR, VI, etc.)', required = False, default = 'CR')
    parser.add_argument('-s', '--sex', help = 'Bird sex (M, F, U))', required = False, default = 'U')
    args = parser.parse_args()
    args.directory = os.path.normpath(args.directory)
    analyze(args.directory, args.uuid, args.age, args.type, args.sex)
