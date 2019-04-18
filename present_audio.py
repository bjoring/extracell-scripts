#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:07:45 2018

"""

"""
    A zmq client to remote control open-ephys GUI
"""

import zmq
import time
import sounddevice as sd
import sudoku as sdk
import stimset as ss
import numpy as np
import argparse
import glob
import json
import pandas as pd
import os
from scipy.io.wavfile import read

def write_log(bird,location,stimset,stimtype,songname,recpath,presentation,seed=None):
    log = {
    "bird":bird,
    "location":location,
    "seed":seed,
    "stimset":stimset,
    "stimtype":stimtype,
    "songs":songname,
    "path":recpath.decode(),
    "presentation":presentation,
    }
    with open('/'.join(recpath.decode().split('/'))+'/'+recpath.decode().split('/')[-1]+'.log','w') as f:
        json.dump(log,f)
    print("Log written to file.")
    
def make_stims(stimset,seed):
    song,Fs,songname = ss.clean(stimset)
    stype = ['continuous','continuous+noise1','continuous+noise2','gap1','gap2','gap+noise1','gap+noise2','noise1','noise2']
    syllables = pd.read_csv('../restoration_syllables.csv')
    conditions = len(stype)
    trials = 10
    size = len(songname)*conditions
    present_order = sdk.seqorder(size,trials)
    presentation = {}
    stims = np.zeros((size,len(song[0])))
    np.random.seed(seed)
    for i in range(len(songname)):    
        ssyll = syllables.loc[syllables.songid==songname[i]]
        sg = np.random.choice(len(ssyll.loc[1:-1]),2,replace=False)
        blocks = np.zeros((2,2))
        blocks[:,0] = np.asarray([list(ssyll.start)[sg[x]] for x in range(2)])
        blocks[:,1] = np.asarray([list(ssyll.end)[sg[x]] for x in range(2)])
        blocks = blocks*Fs
        blocks = blocks.astype(int)
        gsize = int(0.1*Fs)
        wnamp = 25000
        if blocks[0,1]-blocks[0,0] > gsize:
            middle = np.mean(blocks,1)
            blocks[0,0] = middle[0]-int(gsize/2)-50
            blocks[0,1] = blocks[0,0]+gsize
        if blocks[1,1]-blocks[1,0] > gsize:
            middle = np.mean(blocks,1)
            blocks[1,0] = middle[1]-int(gsize/2)-50
            blocks[1,1] = blocks[1,0]+gsize
        N = 50
        ix = np.arange(N*3)
        signal = np.cos(2*np.pi*ix/float(N*2))*0.5+0.5
        fadein = signal[50:100]
        fadeout = signal[0:50]
        for k in range(conditions):
            order = (i*conditions)+k
            presentation[order] = {
                "song":songname[i],
                "type":stype[k],
                }  
            if k == 0:
                stims[order] = song[i]
            elif k==1:
                presentation[order]['gaps'] = blocks[0].tolist()
                wn = np.random.normal(0,wnamp,size=blocks[0,1]-blocks[0,0])
                stims[order] = np.concatenate((song[i][:blocks[0,0]],
                                                 song[i][blocks[0,0]:blocks[0,1]]+wn,
                                                 song[i][blocks[0,1]:]))   
            elif k==2:
                presentation[order]['gaps'] = blocks[1].tolist()
                wn = np.random.normal(0,wnamp,size=blocks[1,1]-blocks[1,0])
                stims[order] = np.concatenate((song[i][:blocks[1,0]],
                                                 song[i][blocks[1,0]:blocks[0,1]]+wn,
                                                 song[i][blocks[1,1]:]))            
            elif k==3:
                presentation[order]['gaps'] = blocks[0].tolist()
                stims[order] = np.concatenate((song[i][:blocks[0,0]],
                                                 song[i][blocks[0,0]:blocks[0,0]+len(fadein)]*fadeout,
                                                 np.zeros(blocks[0,1]-blocks[0,0]-len(fadein)*2),
                                                 song[i][blocks[0,1]-len(fadein):blocks[0,1]]*fadein,
                                                 song[i][blocks[0,1]:]))
            elif k==4:
                presentation[order]['gaps'] = blocks[1].tolist()
                stims[order] = np.concatenate((song[i][:blocks[1,0]],
                                                 song[i][blocks[1,0]:blocks[1,0]+len(fadein)]*fadeout,
                                                 np.zeros(blocks[1,1]-blocks[1,0]-len(fadein)*2),
                                                 song[i][blocks[1,1]-len(fadein):blocks[1,1]]*fadein,
                                                 song[i][blocks[1,1]:]))
            elif k==5:
                presentation[order]['gaps'] = blocks[0].tolist()
                wn = np.random.normal(0,wnamp,size=blocks[0,1]-blocks[0,0])
                stims[order] = np.concatenate((song[i][:blocks[0,0]],
                                                 wn,
                                                 song[i][blocks[0,1]:]))
            elif k==6:
                presentation[order]['gaps'] = blocks[1].tolist()
                wn = np.random.normal(0,wnamp,size=blocks[1,1]-blocks[1,0])
                stims[order] = np.concatenate((song[i][:blocks[1,0]],
                                                 wn,
                                                 song[i][blocks[1,1]:]))
            elif k==7:
                presentation[order]['gaps'] = blocks[0].tolist()
                wn = np.random.normal(0,wnamp,size=blocks[0,1]-blocks[0,0])
                temp = np.zeros(len(song[i]))
                stims[order] = np.concatenate((temp[:blocks[0,0]],
                                                 wn,
                                                 temp[blocks[0,1]:]))
            elif k==8:
                presentation[order]['gaps'] = blocks[1].tolist()
                wn = np.random.normal(0,wnamp,size=blocks[1,1]-blocks[1,0])
                temp = np.zeros(len(song[i]))
                stims[order] = np.concatenate((temp[:blocks[1,0]],
                                                 wn,
                                                 temp[blocks[1,1]:]))
            else:
                print("Undefined stim type")
    scale = np.max(stims)
    pstims = np.zeros((size,len(song[0]),2))
    for i in range(len(pstims)):
        temp = (stims[i]/scale)*(2**15-1)
        pstims[i] = ss.pulsestim(temp)

    return(Fs,stype,present_order,presentation,pstims,songname)

def next_stim(stims,order,i,songname,Fs,stype=None):
    print("Presentation number "+str(i+1)+":")
    if not stype:
        print(songname[(order[i]-1)%len(songname)])
        sd.play(stims[(order[i]-1)%len(songname)],Fs)
        sd.wait()
    else:
        print(songname[(order[i]-1)//len(stype)]+", "+stype[(order[i]-1)%len(stype)])
        sd.play(stims[(order[i]-1)],Fs)
        sd.wait()

def run_gap(bird,location,seed,rec_dir='/home/melizalab/Data',stimset='../Stims1'):

    # Basic start/stop commands
    start_cmd = 'StartRecord'
    stop_cmd = 'StopRecord'
    experiment = 'induction'
    
    print("Bird:",bird)
    print("Recording location:",location)
    print("Randomization seed:",str(seed))
    print("Stimulus directory:",stimset)
    print("Outer directory:", rec_dir)
    
    command = start_cmd + ' RecDir=%s' % rec_dir + ' PrependText=%s' % bird + ' AppendText=%s' % experiment

    # Connect network handler
    ip = '127.0.0.1'
    port = 5556
    timeout = 1.

    url = "tcp://%s:%d" % (ip, port)
    
    Fs,stype,present_order,presentation,stims,songname = make_stims(stimset,seed)
    
    with zmq.Context() as context:
        with context.socket(zmq.REQ) as socket:
            socket.RCVTIMEO = int(timeout * 1000)  # timeout in milliseconds
            socket.connect(url)

            # Start data acquisition
            socket.send_string('StartAcquisition')
            print(socket.recv().decode())
            time.sleep(5)

            socket.send_string('IsAcquiring')
            print("IsAcquiring:", socket.recv().decode())
            print("")
            
            socket.send_string(command)
            print(socket.recv().decode())

            for i in range(len(present_order)):
                
                if i == 0:
                    socket.send_string('IsRecording')
                    print("IsRecording:", socket.recv().decode())
                    socket.send_string('GetRecordingPath')
                    recpath = socket.recv()
                    print("Recording path:", recpath.decode())
                    print("")
                else:
                    socket.send_string(start_cmd)
                    print(socket.recv().decode())
                    
                socket.send_string('IsRecording')
                print("IsRecording:", socket.recv().decode())
                time.sleep(0.5)
                
                next_stim(stims,present_order,i,songname,Fs,stype)
                
                time.sleep(0.5)
                socket.send_string(stop_cmd)
                print(socket.recv().decode())
                socket.send_string('IsRecording')
                print("IsRecording:",socket.recv().decode())
                print("")
                time.sleep(0.5)
                
            time.sleep(0.5)
            
            write_log(bird,location,stimset,experiment,songname,recpath,presentation,seed)
            
            socket.send_string('StopAcquisition')
            print(socket.recv().decode())
    
def run_clean(bird,location,rec_dir='/home/melizalab/Data',stimset='../Stims1'):
    # Basic start/stop commands
    start_cmd = 'StartRecord'
    stop_cmd = 'StopRecord'
    experiment = 'selectivity'
    
    print("Bird:",bird)
    print("Recording location:",location)
    print("Stimulus directory:",stimset)
    print("Outer directory:", rec_dir)
    
    command = start_cmd + ' RecDir=%s' % rec_dir + ' PrependText=%s' % bird + ' AppendText=%s' % stimtype

    # Connect network handler
    ip = '127.0.0.1'
    port = 5556
    timeout = 1.

    url = "tcp://%s:%d" % (ip, port)
    
    songfiles = glob.glob(stimset+'/*.wav')
    size = len(songfiles)
    present_order = sdk.seqorder(size)
    presentation = []
    stim,Fs,songname = ss.clean(stimset)
    pstims = np.zeros((size,len(stim[0]),2))
    for i in range(len(stim)):
        pstims[i] = ss.pulsestim(stim[i])

    with zmq.Context() as context:
        with context.socket(zmq.REQ) as socket:
            socket.RCVTIMEO = int(timeout * 1000)  # timeout in milliseconds
            socket.connect(url)

            # Start data acquisition
            socket.send_string('StartAcquisition')
            print(socket.recv().decode())
            time.sleep(5)

            socket.send_string('IsAcquiring')
            print("IsAcquiring:", socket.recv().decode())
            print("")
            
            socket.send_string(command)
            print(socket.recv().decode())

            for i in range(len(present_order)):
                
                if i == 0:
                    socket.send_string('IsRecording')
                    print("IsRecording:", socket.recv().decode())
                    socket.send_string('GetRecordingPath')
                    recpath = socket.recv()
                    print("Recording path:", recpath.decode())
                    print("")
                else:
                    socket.send_string(start_cmd)
                    print(socket.recv().decode())
                socket.send_string('IsRecording')
                print("IsRecording:", socket.recv().decode())
                time.sleep(1)
                
                next_stim(pstims,present_order,i,songname,Fs)

                time.sleep(1)
                socket.send_string(stop_cmd)
                print(socket.recv().decode())
                socket.send_string('IsRecording')
                print("IsRecording:",socket.recv().decode())
                print("")
                time.sleep(0.5)
            #socket.send_string(stop_cmd)
            #print(socket.recv().decode())
            #socket.send_string('IsRecording')
            #print("IsRecording:", socket.recv().decode())
            # Finally, stop data acquisition; it might be a good idea to
            # wait a little bit until all data have been written to hard drive
            time.sleep(0.5)
            
            write_log(bird,location,stimset,experiment,songname,recpath,presentation)
            
            socket.send_string('StopAcquisition')
            print(socket.recv().decode())
            
def run_chorus(bird,location,seed,rec_dir='/home/melizalab/Data',stimset='../Chorus'):
    # Basic start/stop commands
    start_cmd = 'StartRecord'
    stop_cmd = 'StopRecord'
    experiment = 'chorus'
    
    print("Bird:",bird)
    print("Recording location:",location)
    print("Randomization seed:",str(seed))
    print("Stimulus directory:",stimset)
    print("Outer directory:", rec_dir)
    
    command = start_cmd + ' RecDir=%s' % rec_dir + ' PrependText=%s' % bird + ' AppendText=%s' % experiment

    # Connect network handler
    ip = '127.0.0.1'
    port = 5556
    timeout = 1.

    url = "tcp://%s:%d" % (ip, port)
    
    songfiles = glob.glob('../Chorus/*stim*wav')
    scenefiles = glob.glob('../Chorus/scene*wav')
    songfiles.sort()
    scenefiles.sort()
    
    songname = [os.path.basename(song).strip('.wav') for song in songfiles]
    scenename = [os.path.basename(scene).strip('.wav') for scene in scenefiles]
    songs = []
    scenes = []
    for i in songfiles:
        Fs,s = read(i)
        songs.append(s)
    songs = np.asarray(songs)
    for i in scenefiles:
        Fsc,c = read(i)
        scenes.append(c)
    scenes = np.asarray(scenes)
    
    stype = ['silence','chorus']
    conditions = len(stype)
    size = len(songname)*conditions
    present_order = sdk.seqorder(size)
    presentation = {}
    stim = np.zeros((len(songs),len(scenes[0])))
    scenestim = np.zeros((len(songs)*len(scenes),len(scenes[0])))
    
    for i in range(len(songs)):
        stimstart = (len(scenes[0])-len(songs[i]))//2
        stim[i,stimstart:stimstart+len(songs[i])] = songs[i]
        for j in range(len(scenes)):
            scenestim[(i*len(scenes))+j] = scenes[j] + stim[i]
    
    np.random.seed(seed)
    
    presentation = {}
    pstims = np.zeros((len(songs),len(scenes[0]),2))
    pscenes = np.zeros((len(songs)*len(scenes),len(scenes[0]),2))
    for i in range(len(stim)):
        pstims[i] = ss.pulsestim(stim[i])
    for i in range(len(scenestim)):
        pscenes[i] = ss.pulsestim(scenestim[i])

    with zmq.Context() as context:
        with context.socket(zmq.REQ) as socket:
            socket.RCVTIMEO = int(timeout * 1000)  # timeout in milliseconds
            socket.connect(url)

            # Start data acquisition
            socket.send_string('StartAcquisition')
            print(socket.recv().decode())
            time.sleep(5)

            socket.send_string('IsAcquiring')
            print("IsAcquiring:", socket.recv().decode())
            print("")
            
            socket.send_string(command)
            print(socket.recv().decode())

            for i in range(len(present_order)):
                
                if i == 0:
                    socket.send_string('IsRecording')
                    print("IsRecording:", socket.recv().decode())
                    socket.send_string('GetRecordingPath')
                    recpath = socket.recv()
                    print("Recording path:", recpath.decode())
                    print("")
                else:
                    socket.send_string(start_cmd)
                    print(socket.recv().decode())
                    
                socket.send_string('IsRecording')
                print("IsRecording:", socket.recv().decode())
                time.sleep(0.25)
                
                if present_order[i] <= len(pstims):
                    songnum = present_order[i]-1
                    print("Presentation number "+str(i+1)+": "+songname[songnum])
                    sd.play(pstims[songnum],Fs)
                    sd.wait()
                    presentation[i] = {
                        "song":songname[songnum],
                        "type":"no-scene",
                    }  
                else:
                   scenenum = (present_order[i]-len(songname)-1)+(len(songname)*(i//(len(songname)*2)))
                   songnum = scenenum//10
                   print("Presentation number "+str(i+1)+": "+songname[songnum]+" with chorus")
                   sd.play(pscenes[scenenum],Fsc)
                   sd.wait()
                   presentation[i] = {
                       "song":songname[songnum],
                       "type":scenename[i//(len(songname)*2)],
                   }  
                   
                
                socket.send_string(stop_cmd)
                print(socket.recv().decode())
                socket.send_string('IsRecording')
                print("IsRecording:",socket.recv().decode())
                print("")
                time.sleep(0.25)
                
            time.sleep(0.5)
            
            write_log(bird,location,stimset,experiment,songname,recpath,presentation,seed)
            
            socket.send_string('StopAcquisition')
            print(socket.recv().decode())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment',help='Experiment to run',required=False,default='induction',choices=['induction','selectivity','chorus'])
    parser.add_argument('-b','--bird',help='Bird ID',required=True)
    parser.add_argument('-l','--loc',help='Recording location',required=False)
    parser.add_argument('-s','--seed',help='Seed for syllable randomization',type=int,required=False,default=1)
    parser.add_argument('-d','--dir',help='Recording directory',required=False,default='/home/melizalab/Data')
    parser.add_argument('-ss','--stimset',help='Directory of stimulus set',required=False,default='../Stims1')
    args = parser.parse_args()
    if args.experiment == 'induction':
        run_gap(args.bird,args.loc,args.seed,args.dir,args.stimset)
    elif args.experiment == 'chorus':
        run_chorus(args.bird,args.loc,args.seed,args.dir,args.stimset)
    else:
        run_clean(args.bird,args.loc,args.dir,args.stimset)