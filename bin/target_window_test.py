#target_window_test.py 1/20/2017-3/15/2017
#Timothy Becker, UCONN/SOE/CSE PhD Candidate
#tester script to check the upper window merging algorithms in safe_core.pyx

import argparse
import os
import glob
import time

#----------------------------
import ctypes
import h5py
import math
import numpy as np
import scipy.stats as st
import multiprocessing as mp
import pysam
import safe
import safe_core as core
                  
#set up basic vairables and test sizes
chunk,window,max_depth,FN,v,disjoint = int(1E6),int(1E2),int(1E3),8,30,False
N,SUM,MIN,MAX,M1,M2,M3,M4 = range(FN)

def merge_tiled_spectrum_bin_target(Y,Z,B,t,disjoint=True):
    T = np.zeros((len(B)-1,),dtype=np.float32)
    w = 0
    for i in range(len(B)-1): w += Y[i] 
    b,y,v = len(B)-1,np.uint32(len(Y)/(len(B)-1)),np.uint32(t/w)
    if t%w>1: v += 1
    if v>0 and y>0 and y>=v*w:
        if disjoint:#-------------------------------------
            for i in range(0,y-v,v): #the whole numebr of target windows
                c = i*b
                for j in range(b): T[j] = Y[c+j]
                for j in range(1,v):
                    d = (i+j)*b
                    for k in range(b): T[k] += Y[d+k]
                c = (i/v)*b
                for j in range(b): Z[c+j] = T[j]
            if y%v>0: #left over windows that are less than t for one last window
                i = y-(y%v)
                for j in range(b): T[j] = Y[i*b+j]
                for j in range(1,y%v):      
                    d = (i+j)*b
                    for k in range(b): T[k] += Y[d+k]
                c = (y/v)*b
                for j in range(b): Z[c+j] = T[j]  
        else:#---------------------------------------------------------------------
            for i in range(y-(v-1)):
                c = i*b
                for j in range(b): T[j] = Y[c+j]
                for j in range(1,v):         #now accumulate the tiles into T
                    d = (i+j)*b             #b is the growing tiles
                    for k in range(b): T[k] += Y[d+k]
                for j in range(b): Z[c+j] = T[j]
    return True
    
#data setup here
tiles = chunk/window
if chunk%window>0: tiles += 1
disjoint_tiles = tiles/v
if tiles%v>0: disjoint_tiles += 1

I = mp.Array(ctypes.c_float,chunk,lock=False)
I[:] = np.random.choice(range(max_depth),len(I))

M  = mp.Array(ctypes.c_double,FN*(chunk-window),lock=False)
MZ = mp.Array(ctypes.c_double,FN*(chunk-v*window),lock=False)

S   = mp.Array(ctypes.c_float,FN*(chunk-window),lock=False)
SZ1 = mp.Array(ctypes.c_float,FN*(chunk-v*window),lock=False)

ST   = mp.Array(ctypes.c_float,FN*(tiles),lock=False)
STZ1 = mp.Array(ctypes.c_float,FN*(tiles-v),lock=False)
STZ2 = mp.Array(ctypes.c_float,FN*(disjoint_tiles),lock=False)

T  = mp.Array(ctypes.c_float,FN*FN*(chunk-window-1),lock=False)
TZ = mp.Array(ctypes.c_float, FN*FN*(chunk-v*window-1))

bins = range(0,max_depth,max_depth/FN)+[np.uint32(-1)]
B = mp.Array(ctypes.c_float,bins,lock=False)

s = safe.SAFE()
#testing results here
core.sliding_moments(I,0,chunk,window,window*10,M)
core.sliding_spectrum_bin(I,0,chunk,window,B,S)
core.sliding_transitions_bin(I,0,chunk,window,B,T)

start = time.time()
core.merge_sliding_moments_target(M,MZ,t=v*window)
stop = time.time()

start = time.time()
x = 0
for i in range(tiles):
    core.spectrum_bin(I,x,x+window,B,ST,FN,FN*i)
    x += window
stop = time.time()

start = time.time()
core.merge_tiled_spectrum_bin_target(ST,STZ1,B,t=v*window,disjoint=False)
stop = time.time()

start = time.time()
core.merge_tiled_spectrum_bin_target(ST,STZ2,B,t=v*window,disjoint=True)
stop = time.time()

#additional debuging and testing machinery-------------------------------------------
#tiles = [[window*t,window*(t+1)] for t in range(chunk/window)]
#if chunk%window>0: tiles += [[window*(chunk/window),chunk]]
#Y2 = mp.Array(ctypes.c_double,FN*(len(tiles)),lock=False)
#if   disjoint and len(tiles)%v>0: Z2 = mp.Array(ctypes.c_double,FN*(1+len(tiles)/v),lock=False)
#elif disjoint:                    Z2 = mp.Array(ctypes.c_double,FN*(len(tiles)/v),lock=False)
#else:                             Z2 = mp.Array(ctypes.c_double,FN*(len(tiles)-(v-1)),lock=False)
#for i in range(len(tiles)):
#    core.exact_moments(I,tiles[i][0],tiles[i][1],Y2,i*FN)
#Y,Z = Y2,Z2
#start = time.time()
#core.merge_tiled_moments_target(Y,Z,t=v*window,disjoint=disjoint)
#stop = time.time()
#print('completed in %s sec'%(stop-start))
