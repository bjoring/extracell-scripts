#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:31:06 2018

@author: melizalab
"""

import argparse
import numpy as np
import sudoku as sdk
import h5py
import pandas as pd
import json

def align_audio(fp,log,thresh=1):
    data = h5py.File(fp)
    n = len([x for x in list(data.keys()) if 'rec' in x])
    with open(log) as json_file:
        meta = json.load(json_file)
    present_order = sdk.seqorder(n//10)
    total_start = 0
    df = pd.DataFrame(index=np.arange(0, n), columns = ['rec','song','condition','stim','total_start','total_stop','total_pulse','local_pulse'])
    for i in range(n):
        if len({len(data['rec_0/'+i]) for i in list(data['rec_0'].keys()) if 'channel' in i}) == 1:
            pulse = data['rec_'+str(i)+'/channel37']
            points = np.where(pulse[:]>thresh)[0]
            dist = np.diff(points)
            points = np.delete(points,np.where(dist==1))
            ind = present_order[i]-1
            if len(points) > 1:
                print("Expecting 1 pulse, got "+str(len(points))+". Adjust threshold.")
            else:
                if type(meta['presentation']) == list:
                    song = meta['presentation'][ind].split(' ')[0]
                    condition = ' '.join(meta['presentation'][ind].split(' ')[1:])
                else:
                    song = meta['presentation'][str(ind)]['song']
                    condition = meta['presentation'][str(ind)]['type']
                df.loc[i]=[i,song,condition,ind,total_start,total_start+len(pulse)-1,total_start+points[0],points[0]]
                total_start = total_start+len(pulse)
        else:
            raise ValueError('Not all channels have same length!')
    df.to_csv('/'.join(fp.split('/')[:-1])+'/alignment.csv',index=False)
    data.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',help='File name',required=True)
    parser.add_argument('-l','--log',help='File name of log',required=True)
    parser.add_argument('-t','--thresh',help='Threshold',required=False,default=0.1)
    args = parser.parse_args()
    align_audio(args.file,args.log,args.thresh)
    
