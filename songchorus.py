#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:49:45 2019

@author: melizalab
"""

import glob
import numpy as np
from scipy.io.wavfile import read, write
import os

def rms(y):
    return (np.sqrt(np.mean(y.astype(float)**2)))

def dB(y):
    a0 = 0.00001*(2**15-1)
    return (20*np.log10(rms(y)/a0))

def scale(dB):
    a0 = 0.00001*(2**15-1)
    return (a0*(10**(dB/20)))

np.random.seed(1)
songfiles = glob.glob('../Chorus/*.wav')
stimfiles = [song for song in songfiles if 'chorus' not in song]
chorusfile = glob.glob('../Chorus/chorus*.wav')[0]
songnames = [os.path.basename(song).strip('.wav') for song in stimfiles]
songs = []
for i in stimfiles:
    fs,s = read(i)
    songs.append(s)
fsc,chorus = read(chorusfile)
scalechorus = (chorus/rms(chorus))*scale(63)
songs48 = [(song/rms(song))*scale(48) for song in songs]
songs53 = [(song/rms(song))*scale(53) for song in songs]
songs58 = [(song/rms(song))*scale(58) for song in songs]
songs63 = [(song/rms(song))*scale(63) for song in songs]
songs68 = [(song/rms(song))*scale(68) for song in songs]
songs73 = [(song/rms(song))*scale(73) for song in songs]
songs78 = [(song/rms(song))*scale(78) for song in songs]
scenestarts = np.random.randint(0,len(chorus)-60000,10)
scenes = [chorus[start:start+60000] for start in scenestarts]
scene63 = [(scene/rms(scene))*scale(63) for scene in scenes]

for i in range(len(songnames)):
    write(os.path.join(os.path.dirname(stimfiles[i]),songnames[i]+'stim48.wav'),fs,songs48[i])
    write(os.path.join(os.path.dirname(stimfiles[i]),songnames[i]+'stim53.wav'),fs,songs53[i])
    write(os.path.join(os.path.dirname(stimfiles[i]),songnames[i]+'stim58.wav'),fs,songs58[i])
    write(os.path.join(os.path.dirname(stimfiles[i]),songnames[i]+'stim63.wav'),fs,songs63[i])
    write(os.path.join(os.path.dirname(stimfiles[i]),songnames[i]+'stim68.wav'),fs,songs68[i])
    write(os.path.join(os.path.dirname(stimfiles[i]),songnames[i]+'stim73.wav'),fs,songs73[i])
    write(os.path.join(os.path.dirname(stimfiles[i]),songnames[i]+'stim78.wav'),fs,songs78[i])

for i in range(len(scene63)):
    write(os.path.join(os.path.dirname(chorusfile),'scene63_'+str(i)+'.wav'),fsc,scene63[i])