#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:29:30 2019

@author: melizalab
"""

import numpy as np
import argparse
from shutil import copy2
import os
import glob
from mountainlab_pytools import mdaio

def preprocess(fp):
    mfiles = os.listdir('../mountain_files')
    [copy2(os.path.join('../mountain_files',file),fp) for file in mfiles]
    rawdat = glob.glob(os.path.join(fp,'*.dat'))[0]
    raw = np.fromfile(rawdat,dtype=np.int16)
    raw = np.reshape(raw,(32,-1),order='F')
    mdaio.writemda16i(raw,os.path.join(fp,'raw.mda'))
    os.remove(rawdat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory',help='Directory',required=True)
    args = parser.parse_args()
    args.directory = os.path.normpath(args.directory)
    preprocess(args.directory)
    
    