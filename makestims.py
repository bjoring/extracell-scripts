# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:08:28 2019

@author: Margot
"""

import sudoku as sdk
import stimset as ss
import numpy as np
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import read, write

def rms(y):
    return (np.sqrt(np.mean(y.astype(float)**2)))

def dB(y, cal = 0.00001*(2**15-1)):
    a0 = cal
    return (20*np.log10(rms(y)/a0))

def scale(dB, cal = 0.00001*(2**15-1)):
    a0 = cal
    return (a0*(10**(dB/20)))

def make_stims(stimset,seed):
    if '180B' in stimset or '178B' in stimset:
        song,Fs,songname = ss.dBclean('../178B')
        song2,Fs,songname2 = ss.dBclean('../180B')
        song = np.concatenate((song,song2),0)
        songname = songname+songname2
    else:
        song,Fs,songname = ss.dBclean(stimset)
    stype = ['continuous','continuousnoise1','continuousnoise2','gap1','gap2','gapnoise1','gapnoise2','noise1','noise2']
    syllables = pd.read_csv('../gap-locations.csv')
    conditions = len(stype)
    trials = 10
    size = len(songname)*conditions
    present_order = sdk.seqorder(size,trials)
    presentation = {}
    stims = []
    np.random.seed(seed)
    for i in range(len(songname)):    
        ssyll = syllables.loc[syllables.song==songname[i]]
        #sg = np.random.choice(ssyll.iloc[1:-1].index,2,replace=False)
        blocks = np.zeros((2,2))
        #print(ssyll.start1.values[0])
        #print(ssyll.start2)
        blocks[:,0] = np.asarray((ssyll.start1.values[0],ssyll.start2.values[0]))
        blocks[:,1] = np.asarray((ssyll.end1.values[0],ssyll.end2.values[0]))
        blocks = blocks*Fs
        blocks = blocks.astype(int)
        gsize = int(0.1*Fs)
        wnamp = 25000
        wndB = 70
        if blocks[0,1]-blocks[0,0] > gsize:
            #middle = np.mean(blocks,1)
            #blocks[0,0] = middle[0]-int(gsize/2)-50
            blocks[0,1] = blocks[0,0]+gsize
        if blocks[1,1]-blocks[1,0] > gsize:
            #middle = np.mean(blocks,1)
            #blocks[1,0] = middle[1]-int(gsize/2)-50
            blocks[1,1] = blocks[1,0]+gsize
        N = 120
        ix = np.arange(N*3)
        signal = np.cos(2*np.pi*ix/float(N*2))*0.5+0.5
        fadein = signal[120:240]
        fadeout = signal[0:120]
        wn1 = np.random.normal(0,wnamp,size=blocks[0,1]-blocks[0,0])
        wn1 = (wn1/rms(wn1))*scale(wndB)
        wn2 = np.random.normal(0,wnamp,size=blocks[1,1]-blocks[1,0])
        wn2 = (wn2/rms(wn2))*scale(wndB)
        mask = np.random.normal(0,wnamp,size=len(song[i]))
        mask = (mask/rms(mask))*scale(wndB)
        mN = 1000
        mix = np.arange(mN*3)
        msignal = np.cos(2*np.pi*mix/float(mN*2))*0.5+0.5
        mfadein = msignal[1000:2000]*np.random.normal(0,wnamp,size=1000)
        mfadein = (mfadein/rms(mfadein))*scale(wndB)
        mfadeout = msignal[0:1000]*np.random.normal(0,wnamp,size=1000)
        mfadeout = (mfadeout/rms(mfadeout))*scale(wndB)
        for k in range(conditions):
            order = (i*conditions)+k
            presentation[order] = {
                "song":songname[i],
                "type":stype[k],
                }  
            if k == 0:
                stims.append(song[i])
            elif k==1:
                presentation[order]['gaps'] = blocks[0].tolist()
                stims.append(np.concatenate((song[i][:blocks[0,0]],
                                                 song[i][blocks[0,0]:blocks[0,1]]+wn1,
                                                 song[i][blocks[0,1]:])))   
            elif k==2:
                presentation[order]['gaps'] = blocks[1].tolist()
                stims.append(np.concatenate((song[i][:blocks[1,0]],
                                                 song[i][blocks[1,0]:blocks[1,1]]+wn2,
                                                 song[i][blocks[1,1]:])))         
            elif k==3:
                presentation[order]['gaps'] = blocks[0].tolist()
                stims.append(np.concatenate((song[i][:blocks[0,0]],
                                                 song[i][blocks[0,0]:blocks[0,0]+len(fadein)]*fadeout,
                                                 np.zeros(blocks[0,1]-blocks[0,0]-len(fadein)*2),
                                                 song[i][blocks[0,1]-len(fadein):blocks[0,1]]*fadein,
                                                 song[i][blocks[0,1]:])))
            elif k==4:
                presentation[order]['gaps'] = blocks[1].tolist()
                stims.append(np.concatenate((song[i][:blocks[1,0]],
                                                 song[i][blocks[1,0]:blocks[1,0]+len(fadein)]*fadeout,
                                                 np.zeros(blocks[1,1]-blocks[1,0]-len(fadein)*2),
                                                 song[i][blocks[1,1]-len(fadein):blocks[1,1]]*fadein,
                                                 song[i][blocks[1,1]:])))
            elif k==5:
                presentation[order]['gaps'] = blocks[0].tolist()
                stims.append(np.concatenate((song[i][:blocks[0,0]],
                                                 wn1,
                                                 song[i][blocks[0,1]:])))
            elif k==6:
                presentation[order]['gaps'] = blocks[1].tolist()
                stims.append(np.concatenate((song[i][:blocks[1,0]],
                                                 wn2,
                                                 song[i][blocks[1,1]:])))
            elif k==7:
                presentation[order]['gaps'] = blocks[0].tolist()
                temp = np.zeros(len(song[i]))
                stims.append(np.concatenate((temp[:blocks[0,0]],
                                                 wn1,
                                                 temp[blocks[0,1]:])))
            elif k==8:
                presentation[order]['gaps'] = blocks[1].tolist()
                temp = np.zeros(len(song[i]))
                stims.append(np.concatenate((temp[:blocks[1,0]],
                                                 wn2,
                                                 temp[blocks[1,1]:])))
            elif k==9:
                stims.append(np.concatenate((mfadein,
                                                song[i]+mask,
                                                mfadeout)))
            elif k==10:
                stims.append(np.concatenate((mfadein,
                                                song[i][:blocks[1,0]]+mask[:blocks[1,0]],
                                                song[i][blocks[1,0]:blocks[1,0]+len(fadein)]*fadeout+mask[blocks[1,0]:blocks[1,0]+len(fadein)],
                                                mask[blocks[1,0]+len(fadein):blocks[1,1]-len(fadein)],
                                                song[i][blocks[1,1]-len(fadein):blocks[1,1]]*fadein+mask[blocks[1,1]-len(fadein):blocks[1,1]],
                                                song[i][blocks[1,1]:]+mask[blocks[1,1]:],
                                                mfadeout)))
            else:
                print("Undefined stim type")
    stims = np.asarray(stims)
    #scale = np.max(stims)
    #pstims = np.zeros((size,len(song[0]),2))
    #for i in range(len(pstims)):
        #temp = (stims[i]/scale)*(2**15-1)
        #pstims[i] = ss.pulsestim(temp)
        #pstims[i] = ss.pulsestim(stims[i])

    return(Fs,stype,present_order,presentation,stims,songname)
    
def write_stims(Fs,presentation,stims):
    for i in range(len(stims)):
        name = presentation[i]['song']+'_'+presentation[i]['type']
        stim = stims[i].astype(np.int16)
        write('../Recordings/Songs/'+name+'.wav', Fs, stim)
    
Fs,stype,present_order,presentation,stims,songname = make_stims('../178B',1)
#print(presentation)
#sd.play(stims[1]/(2**15-1),Fs)
write_stims(Fs,presentation,stims)
