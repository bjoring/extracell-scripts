#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:27:26 2018

@author: melizalab
"""

import zmq
import time
import sounddevice as sd
import sudoku as sdk
import stimset as ss
import argparse
import glob

def search(stimset='../Stims1'):

    # Connect network handler
    ip = '127.0.0.1'
    port = 5556
    timeout = 1.

    url = "tcp://%s:%d" % (ip, port)
    
    songfiles = glob.glob(stimset+'/*.wav')
    size = len(songfiles)
    present_order = sdk.seqorder(size)
    stim,Fs,songname = ss.clean(stimset)
    print(songname)
    
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
            print("Playing search stimuli...")
            print("To end, press CTRL-C")
            
            try:
                while True:
                    for i in range(len(present_order)):
                        sd.play(stim[present_order[i]-1],Fs)
                        sd.wait()
                        time.sleep(2)
            except KeyboardInterrupt:
                print("")
                socket.send_string('StopAcquisition')
                print(socket.recv().decode())
                time.sleep(1)
                socket.send_string('IsAcquiring')
                print("IsAcquiring:", socket.recv().decode())
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss','--stimset',help='Directory of stimulus set',required=False,default='../Stims1')
    args = parser.parse_args()
    search(args.stimset)