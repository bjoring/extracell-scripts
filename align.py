#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:31:06 2018

@author: melizalab
"""

import argparse
import numpy as np
import sudoku as sdk
import OpenEphys

def align_audio(fp,n,thresh=0.1):
    present_order = sdk.seqorder(n)
    align = np.zeros((len(present_order),2))
    pulse = OpenEphys.load(fp)
    points = np.where(pulse['data']>thresh)[0]
    dist = np.diff(points)
    points = np.delete(points,np.where(dist==1))
    if len(points) != len(present_order):
        print("Expecting "+str(len(present_order))+" elements, got "+str(len(points))+". Adjust threshold.")
    else:
        align[:,0] = present_order
        align[:,1] = points
        align = align[align[:,0].argsort()]
        np.savetxt('/'.join(fp.split('/')[:-1])+'/alignment.csv',align,delimiter=',')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',help='File name',required=True)
    parser.add_argument('-n','--nstims',help='Number of Stims',type=int,required=False,default=5)
    parser.add_argument('-t','--thresh',help='Threshold',required=False,default=0.1)
    args = parser.parse_args()
    align_audio(args.file,args.nstims,args.thresh)
    
