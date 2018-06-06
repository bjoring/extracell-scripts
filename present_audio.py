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
import os
import pandas as pd

def run_gap(bird,location,seed,rec_dir='/home/melizalab/Data',stimset='../Stims1'):

    # Basic start/stop commands
    start_cmd = 'StartRecord'
    stop_cmd = 'StopRecord'
    stimtype = 'gap'
    
    print("Bird:",bird)
    print("Recording location:",location)
    print("Randomization seed:",str(seed))
    print("Stimulus directory:",stimset)
    print("Outer directory:", rec_dir)
    
    log = {
            "bird":bird,
            "location":location,
            "seed":seed,
            "stimset":stimset,
            "stimtype":stimtype,
            }
    
    command = start_cmd + ' RecDir=%s' % rec_dir + ' PrependText=%s' % bird + ' AppendText=%s' % stimtype

    # Connect network handler
    ip = '127.0.0.1'
    port = 5556
    timeout = 1.

    url = "tcp://%s:%d" % (ip, port)
    
    songfiles = glob.glob(stimset+'/*.wav')
    size = len(songfiles)
    seed = 1
    present_order = sdk.seqorder(size)
    np.random.seed(seed)
    x = np.zeros((size,len(present_order)//size))
    x[:,len(x[0])//2:len(x[0])] = 1
    [np.random.shuffle(x[row]) for row in range(len(x))]
    
    stim,Fs,songname = ss.gap(seed,stimset)
    log['songs'] = songname
    stype = ['gap','gap + noise']
    pairs = np.zeros((len(present_order),2))
    presentation = []
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
                print("Presentation number "+str(i+1))
                pairs[i,0] = present_order[i]-1
                pairs[i,1] = x[present_order[i]-1,i//10]
                presentation.append(songname[int(pairs[i,0])]+" "+stype[int(pairs[i,1])])
                print("Presenting "+songname[int(pairs[i,0])]+" with "+stype[int(pairs[i,1])])
                sd.play(stim[int(pairs[i,0])*2+int(pairs[i,1])],Fs)
                sd.wait()
                time.sleep(2)
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
            log['path'] = recpath.decode()
            log['presentation'] = presentation
            log['pairs'] = pairs.tolist()
            gaps = pd.read_csv('../gaps.csv')
            gaps = gaps.to_dict('list')
            log['gaps'] = gaps
            os.remove('../gaps.csv')
            socket.send_string('StopAcquisition')
            print(socket.recv().decode())
            with open('../Logs/'+recpath.decode().split('/')[-1]+'.log','w') as f:
                json.dump(log,f)
    
def run_clean(bird,location,rec_dir='/home/melizalab/Data',stimset='../Stims1'):
    # Basic start/stop commands
    start_cmd = 'StartRecord'
    stop_cmd = 'StopRecord'
    stimtype = 'clean'
    
    print("Bird:",bird)
    print("Recording location:",location)
    print("Stimulus directory:",stimset)
    print("Outer directory:", rec_dir)
    
    log = {
            "bird":bird,
            "location":location,
            "stimset":stimset,
            "stimtype":stimtype,
            }
    
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
    log['songs'] = songname
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
                print("Presentation number "+str(i+1))
                presentation.append(songname[present_order[i]-1])
                print("Presenting "+songname[present_order[i]-1])
                sd.play(stim[present_order[i]-1],Fs)
                sd.wait()
                time.sleep(2)
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
            log['path'] = recpath.decode()
            log['presentation'] = presentation
            socket.send_string('StopAcquisition')
            print(socket.recv().decode())
            with open('../Logs/'+recpath.decode().split('/')[-1]+'.log','w') as f:
                json.dump(log,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--type',help='Type of stimulus presentation',required=False,default='gap',choices=['gap','clean'])
    parser.add_argument('-b','--bird',help='Bird ID',required=True)
    parser.add_argument('-l','--loc',help='Recording location',required=False)
    parser.add_argument('-s','--seed',help='Seed for syllable randomization',type=int,required=False,default=1)
    parser.add_argument('-d','--dir',help='Recording directory',required=False,default='/home/melizalab/Data')
    parser.add_argument('-ss','--stimset',help='Directory of stimulus set',required=False,default='../Stims1')
    args = parser.parse_args()
    if args.type == 'gap':
        run_gap(args.bird,args.loc,args.seed,args.dir,args.stimset)
    else:
        run_clean(args.bird,args.loc,args.dir,args.stimset)