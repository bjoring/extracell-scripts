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

def analyze(bp):
    recordings = os.listdir(bp)
    for i in recordings:
        fp = os.path.join(bp,i)
        clusters = glob.glob(os.path.join(fp,'ch*c*'))
        for cluster in clusters:
            spikey(cluster,fp)
            
def spikey(cluster,fp):
    toes = glob.glob(os.path.join(cluster,'*.toe_lis'))
    toes.sort()
    syll = pd.read_csv('../restoration_syllables.csv')
    logfile = glob.glob(os.path.join(fp,'*.log'))
    log = json.load(logfile)
    for toe in toes:
        with open(toes) as fl:
            data = tl.read(fl)
        
        base = os.path.splitext(os.path.basename(toe))[0]
        song,vers = base.split('_')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory',help='Base directory',required=True)
    args = parser.parse_args()
    analyze(args.directory)
