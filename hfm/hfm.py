#Copyright (C) 2019 Timothy James Becker
#OO class that enables OOC || feature extraction via shared memory buffers R/W into hdf5 stores
#arrays are all assumed to be flat ctypes for use with multiprocessing without a GIL
#workflow is to have the chroms you want, some parameters for analysis and then
#you check that the hdf5 store has been generated and init the store before || execution of chucks
#non reads aka non count based tracks don't make sense for spectrum and transitions.
#extraction with 1-bp sliding mode with W=100bp, TN==SN=20 is ~ 140GB per SM/RG
#extraction with tiled mode        with W=100bp, TN==SN=20 is ~ 2GB   per SM/RG 

import os
import ctypes
from h5py import File
import json
import math
import glob
import time
import itertools as it
import numpy as np
from scipy import signal
import multiprocessing as mp
import pysam
import core

#alignemnt file meta-data utilties-------------------------------------------------------------------

#given an alignment file, SAM,BAM,CRAM reads the sequences and creates a name to length dict
#mapping the name and size of each sequence that will be passed to the extractor
#common workflows are to remove the sequences or match to the sequences that are desired
#before passing into a feature extraction runner
def get_sam_seq(alignment_path, sort_by_largest=True):
    am = pysam.AlignmentFile(alignment_path,'rb')
    seqs = {s['SN']:s['LN'] for s in am.header['SQ']}
    am.close()
    return sorted([{seqs[k]:k} for k in seqs], key=lambda x: list(x.keys())[0], reverse=sort_by_largest)

#given an alignment file, SAM, BAM,CRAM reads the sequences and creates a dict mapping:
#{sample_name:[rg1_name,rg2_name,...rgx_name}
def get_sam_rg(alignment_path):
    rg = {}
    am = pysam.AlignmentFile(alignment_path,'rb')
    #need to assemble all the read groups in a file here
    if 'RG' in am.header:
        for i in am.header['RG']:
            if i['SM'] in rg:
                rg[i['SM']] += [i['ID']]
            else:
                rg[i['SM']] =  [i['ID']]
    am.close()
    return rg
    
#given an alignment file, SAM, BAM,CRAM reads the sequences and creates a dict mapping:
#{rg1_name:sample_id}
def get_sam_sm(alignment_path):
    sm = {}
    am = pysam.AlignmentFile(alignment_path,'rb')
    if 'RG' in am.header:
        for i in am.header['RG']:
            sm[i['ID']] = i['SM']
    am.close()
    return sm

#sm->rg->seq->
def get_hdf5_sm(hdf5_path):
#    sm = {}
#    f = h5py.File(hdf5_path, 'a')
    return True

def get_hdf5_map(hdf5_path):
    S = {}
    f = File(hdf5_path,'r')
    for sm in f:
        S[sm] = {}
        for rg in f[sm]:
            S[sm][rg] = {}
            for seq in f[sm][rg]:
                S[sm][rg][seq] = {}
                for trk in f[sm][rg][seq]:
                    S[sm][rg][seq][trk] = {}
                    for ftr in f[sm][rg][seq][trk]:
                        S[sm][rg][seq][trk][ftr] = sorted(list(f[sm][rg][seq][trk][ftr].keys()),key=lambda x: int(x))
    return S

def get_hdf5_windows(hdf5_path,check=True):
        W,same = {},{}
        if os.path.exists(hdf5_path):
            if check: print('checking file %s'%hdf5_path)
            f = File(hdf5_path,'r')
            if len(f.keys())>0:
                sm = list(f.keys())[0]
                if len(f[sm].keys())>0:
                    if check: print('found sm %s'%sm)
                    rg = list(f[sm].keys())[0]
                    if len(f[sm][rg].keys())>0:
                        for seq in f[sm][rg]:
                            if seq not in W: W[seq] = []
                            if len(f[sm][rg][seq].keys())>0:
                                trk = list(f[sm][rg][seq].keys())[0]
                                if len(f[sm][rg][seq][trk].keys())>0:
                                    ftr = list(f[sm][rg][seq][trk].keys())[0]
                                    if len(f[sm][rg][seq][trk][ftr].keys())>0:
                                        if len(W[seq])<=0: W[seq] = sorted([int(x) for x in list(f[sm][rg][seq][trk][ftr].keys())])
                                        w = sorted([int(x) for x in list(f[sm][rg][seq][trk][ftr].keys())])
                                        if check and w!=W[seq]: same[(rg,seq,trk,ftr)] = w
            f.close()
            if check and len(same)>0:
                W = []
                print('non-uniform window lengths detected at [sm][rg][seq][trk][ftr] level:%s\n'%same)
            elif check and len(same)<=0:
                print('all window lengths for sm %s are equal'%sm)
        return W

def get_hdf5_attrs(hdf5_path):
    S = {}
    f = File(hdf5_path,'r')
    for sm in f:
        S[sm] = {}
        for rg in f[sm]:
            S[sm][rg] = {}
            for seq in f[sm][rg]:
                S[sm][rg][seq] = {}
                trk = list(f[sm][rg][seq].keys())[0]
                ftr = list(f[sm][rg][seq][trk].keys())[0]
                for w in f[sm][rg][seq][trk][ftr]:
                    S[sm][rg][seq][w] = {}
                    for k in f[sm][rg][seq][trk][ftr][w].attrs.keys():
                        S[sm][rg][seq][w][k] = f[sm][rg][seq][trk][ftr][w].attrs[k]
    return S

#given a bunch of hdf5s done in ||, merge into one
#may have to midfiy for multi window version...
def merge_seqs(hdf5_dir, hdf5_out):
    hdf5_files = glob.glob(hdf5_dir+'/*.hdf5')
    print('merging the following files: %s'%hdf5_files)
    out_f = File(hdf5_out,'a')
    for hdf5_file in hdf5_files:
        print('reading %s'%hdf5_file)
        in_f = File(hdf5_file,'r')
        for sample in in_f:
            print('reading sample %s'%sample)
            for rg in in_f[sample]:
                for seq in in_f[sample][rg]:
                    for track in in_f[sample][rg][seq]:
                        for feature in in_f[sample][rg][seq][track]:
                            g_path = '/'.join([sample,rg,seq,track,feature]) #/sample/rg/seq/track: features...
                            g_id = out_f.require_group('/'.join([sample,rg,seq,track]))
                            print('copying %s file\n\tg_path=%s'%(hdf5_file,g_path))
                            in_f.copy(g_path,g_id,name=feature)
        in_f.close()
    out_f.close()
    return True

def merge_samples(hdf5_dir,hdf5_out):
    hdf5_files = glob.glob(hdf5_dir+'/*.hdf5')
    print('merging the following files: %s'%hdf5_files)
    out_f = File(hdf5_out,'a')
    for hdf5_file in hdf5_files:
        print('reading %s'%hdf5_file)
        in_f = File(hdf5_file,'r')
        for sample in in_f:
            print('reading sample %s'%sample)
            for rg in in_f[sample]:
                for seq in in_f[sample][rg]:
                    for track in in_f[sample][rg][seq]:
                        for feature in in_f[sample][rg][seq][track]:
                            g_path = '/'.join([sample,rg,seq,track,feature]) #/sample/rg/seq/track: features...
                            g_id = out_f.require_group('/'.join([sample,rg,seq,track]))
                            print('copying %s file\n\tg_path=%s'%(hdf5_file,g_path))
                            in_f.copy(g_path,g_id,name=feature)
        in_f.close()
    out_f.close()
    return True

#main sequence alignment feature extraction class  
#add some metadata information about the sequencing platform PL
#add some metadata information abut the type of data: WGS, WES, RNA-seq, CHIAPET, etc... 
class HFM:
    def __init__(self,window=int(1E2),window_branch=0,window_root=0,chunk=int(1E6),max_depth=int(1E4),
                 bins=None,tile=True,fast=True,linear=False,compression='gzip',ratio=9,
                 fr_path=None,gw_path=None):
        self.__max_depth__     = np.uint32(max_depth)     #saturate counts after this value
        self.__window__        = np.uint32(window)        #base window size
        self.__window_branch__ = np.uint32(window_branch) #branch factor each level
        self.__window_root__   = np.uint32(window_root)   #root window size
        self.__chunk__     = np.uint32(chunk)     #chunk size used for OOC memory
        self.__tile__      = tile                 #only tiles are stored
        self.__linear__    = linear               #flat array implies multiprocessing with ctypes
        self.A  = None                            #the recarray attach point
        self.I  = None                            #the input shared memory buffer
        self.O  = None                            #the output shared memory buffer
        self.ap = None                            #pysam alignment path
        self.f  = None                            #hdf5 file pointer
        self.g  = None                            #secondary hdf5 file pointer if needed
        self.__restart__ = 4                      #restart divisor of window if using fast algo
        self.__compression__ = compression        #can use lzf for fast float or gzip here
        self.__ratio__ = ratio                    #comp ratio 1-9
        if not bins is None:                      #bins used to generate counts
            self.__bins__ = bins                  #user definable
        else: #can add different bin functions here later...
            self.__bins__ = [x for x in range(0,self.__max_depth__,np.uint32(self.__max_depth__/int(1E1)))]+[np.uint32(-1)]
        self.B = mp.Array(ctypes.c_float,self.__bins__,lock=False)
        self.__fast__ = fast #fast implies that you are doing a sliding window calculation
        #moment positions for easy future additions
        self.__N__,self.__SUM__,self.__MIN__,self.__MAX__, \
        self.__M1__,self.__M2__,self.__M3__,self.__M4__ = [x for x in range(8)]
        self.__MN__,self.__SN__,self.__TN__ = 8,len(self.__bins__)-1,len(self.__bins__)-1
        self.__moments__  = ['N','SUM','MIN','MAX','M1','M2','M3','M4']
        self.__tracks__   = ['total','proper_pair','discordant','primary','alternate',
                             'orient_same','orient_out','orient_um','orient_chr',
                             'clipped','deletion','insertion','substitution','fwd_rev_diff',
                             'mapq','mapq_pp','mapq_dis','tlen','tlen_pp','tlen_dis','GC']
        self.__features__ = ['moments','spectrum','transitions']
        self.__buffer__ = None

    def __enter__(self):
        return self

    def __del__(self):
        return 0
        
    def __exit__(self, type, value, traceback):
        return 0
    
        #A is a numpy recarray, F is a list of fields or columns to slice out of the array
    def view(self,A,F):
       return np.ndarray(A.shape,np.dtype({name:A.dtype.F[name] for name in F}),A,0,A.strides)  
    
    #store the sms information too?
    def write_attrs(self,data):
        data.attrs['N']             = self.__N__
        data.attrs['SUM']           = self.__SUM__
        data.attrs['MIN']           = self.__MIN__
        data.attrs['MAX']           = self.__MAX__
        data.attrs['M1']            = self.__M1__
        data.attrs['M2']            = self.__M2__
        data.attrs['M3']            = self.__M3__
        data.attrs['M4']            = self.__M4__
        data.attrs['MN']            = self.__MN__
        data.attrs['SN']            = self.__SN__
        data.attrs['TN']            = self.__TN__
        data.attrs['seq']           = self.__seq__
        data.attrs['len']           = self.__len__
        data.attrs['fast']          = self.__fast__
        data.attrs['linear']        = self.__linear__
        data.attrs['tile']          = self.__tile__
        data.attrs['window']        = self.__window__
        data.attrs['window_branch'] = self.__window_branch__
        data.attrs['window_root']   = self.__window_root__
        data.attrs['moments']       = self.__moments__
        data.attrs['bins']          = self.__bins__
        data.attrs['max_depth']     = self.__max_depth__
        data.attrs['buffer']        = self.__buffer__
        #data.attrs['pos']           = self.__pos__

    #store the sms information too?
    def read_attrs(self,data):
        self.__N__              = data.attrs['N']
        self.__SUM__            = data.attrs['SUM']
        self.__MIN__            = data.attrs['MIN']
        self.__MAX__            = data.attrs['MAX']
        self.__M1__             = data.attrs['M1']
        self.__M2__             = data.attrs['M2']
        self.__M3__             = data.attrs['M3']
        self.__M4__             = data.attrs['M4']
        self.__MN__             = data.attrs['MN']
        self.__SN__             = data.attrs['SN']
        self.__TN__             = data.attrs['TN']
        self.__seq__            = data.attrs['seq']
        self.__len__            = data.attrs['len']
        self.__fast__           = data.attrs['fast']
        self.__linear__         = data.attrs['linear']
        self.__tile__           = data.attrs['tile']
        self.__window__         = data.attrs['window']
        self.__window_branch__  = data.attrs['window_branch']
        self.__window_root__    = data.attrs['window_root']
        self.__moments__        = data.attrs['moments']
        self.__bins__           = data.attrs['bins']
        self.__max_depth__      = data.attrs['max_depth']
        self.__buffer__         = data.attrs['buffer']
        #self.__pos__            = data.attrs['pos']
        self.B = mp.Array(ctypes.c_float,self.__bins__,lock=False)

    def set_attributes(self,dict):
        self.__N__              = dict['N']
        self.__SUM__            = dict['SUM']
        self.__MIN__            = dict['MIN']
        self.__MAX__            = dict['MAX']
        self.__M1__             = dict['M1']
        self.__M2__             = dict['M2']
        self.__M3__             = dict['M3']
        self.__M4__             = dict['M4']
        self.__MN__             = dict['MN']
        self.__SN__             = dict['SN']
        self.__TN__             = dict['TN']
        self.__seq__            = dict['seq']
        self.__len__            = dict['len']
        self.__fast__           = dict['fast']
        self.__linear__         = dict['linear']
        self.__tile__           = dict['tile']
        self.__window__         = dict['window']
        self.__window_branch__  = dict['window_branch']
        self.__window_root__    = dict['window_root']
        self.__moments__        = dict['moments']
        self.__bins__           = dict['bins']
        self.__max_depth__      = dict['max_depth']
        self.__buffer__         = dict['buffer']
        #self.__pos__            = data.attrs['pos']
        self.B = mp.Array(ctypes.c_float,self.__bins__,lock=False)

    def get_attributes(self):
        dict = {}
        dict['N']             = self.__N__
        dict['SUM']           = self.__SUM__
        dict['MIN']           = self.__MIN__
        dict['MAX']           = self.__MAX__
        dict['M1']            = self.__M1__
        dict['M2']            = self.__M2__
        dict['M3']            = self.__M3__
        dict['M4']            = self.__M4__
        dict['MN']            = self.__MN__
        dict['SN']            = self.__SN__
        dict['TN']            = self.__TN__
        dict['seq']           = self.__seq__
        dict['len']           = self.__len__
        dict['fast']          = self.__fast__
        dict['linear']        = self.__linear__
        dict['tile']          = self.__tile__
        dict['window']        = self.__window__
        dict['window_branch'] = self.__window_branch__
        dict['window_root']   = self.__window_root__
        dict['moments']       = self.__moments__
        dict['bins']          = self.__bins__
        dict['max_depth']     = self.__max_depth__
        dict['buffer']        = self.__buffer__
        # data.attrs['pos']   = self.__pos__
        return dict

    #given an integer range pull the corresponding tiles
    #because disjoint tiles will partially overlap, this
    #can return tiles before and after the range selected
    def range_to_tiles(self):
        return []

    #given a tile which has start
    def tile_to_index(self,tile):
        return 0
        
    #default track is reads_all or total coverage from an alignment file
    #into the shared memory buffers, perform feature generation
    #and then write write to hdf5 container with extraction metadata
    def extract_chunk(self, alignment_path, hdf5_path, sms, seq, start, end, 
                      merge_rg=True,tracks=['total'],features=['moments'],verbose=False):
        #PRE------------------------------------------------------------------------------------------------------PRE 
        self.__seq__ = list(seq.keys())[0]
        self.__len__ = seq[list(seq.keys())[0]]
        self.ap = alignment_path
        if merge_rg: sms = {'all':'-'.join(sorted(list(set(sms.values()))))}
        if len(tracks)==1 and tracks[0]=='all':     tracks = self.__tracks__
        if len(features)==1 and features[0]=='all': features = self.__features__
        end = min(end,seq[list(seq.keys())[0]]) #correct any loads that are out of bounds
        self.A = core.load_reads_all_tracks(self.ap,sms,self.__seq__,start,end,merge_rg) #dict, single loading
        self.f = File(hdf5_path, 'a')
        #PRE------------------------------------------------------------------------------------------------------PRE
        #TILES--------------------------------------------------------------------------------------------------TILES
        if self.__tile__:
            #[1] set the chunk tiles including the last partial
            tiles = [[i,i+self.__window__] for i in range(0,end-start,self.__window__)] #full windows
            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
            #[2] set the total container size in case it is not intialized
            t = int(self.__len__/self.__window__)
            if  self.__len__%self.__window__>0: t += 1
            #[3] iterate on the tracks return from the input buffer tracks
            for track in tracks:
                if track in self.A:
                    for rg in self.A[track]:
                        self.I = self.A[track][rg] #pull out the track here-----------------------------------------------
                        if 'moments' in features:
                            self.__buffer__ = 'moments'
                            if not sms[rg]+'/'+rg+'/'+list(seq.keys())[0]+'/'+track+'/moments/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                                if self.__linear__:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     (t*self.__MN__,),dtype='f8',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     (t*self.__MN__,),dtype='f8',shuffle=True)
                                else:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     (t,self.__MN__),dtype='f8',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     (t,self.__MN__),dtype='f8',compression=self.__compression__,shuffle=True)
                            else: #need to delete the entry?
                                data = self.f[sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__]
                            self.O = mp.Array(ctypes.c_double,self.__MN__*len(tiles),lock=False)
                            for i in range(len(tiles)):                                                #make ||
                                core.exact_moments(self.I,tiles[i][0],tiles[i][1],self.O,i*self.__MN__)  #make ||
                            a,b = int(start/self.__window__),int(start/self.__window__+len(tiles))
                            if self.__linear__:
                                data[a*self.__MN__:b*self.__MN__] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(int(len(self.O)/self.__MN__),self.__MN__))
                                data[a:b,:] = self.O[:,:]
                            self.write_attrs(data)
                        if 'spectrum' in features:
                            self.__buffer__ = 'spectrum'
                            if not sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                                if self.__linear__:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                     (t*self.__SN__,),dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                     (t*self.__SN__,),dtype='f4',compression=self.__compression__,shuffle=True)
                                else:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                     (t,self.__SN__),dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                     (t,self.__SN__),dtype='f4',compression=self.__compression__,shuffle=True)
                            else: #need to delete the entry?
                                data = self.f[sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__]
                            self.O = mp.Array(ctypes.c_float,self.__SN__*len(tiles),lock=False)
                            for i in range(len(tiles)):                                               #make ||
                                core.spectrum_bin(self.I,tiles[i][0],tiles[i][1],                       #make ||
                                                self.B,self.O,self.__SN__,i*self.__SN__)              #make ||
                            a,b = int(start/self.__window__),int(start/self.__window__+len(tiles))
                            if self.__linear__:
                                data[a*self.__SN__:b*self.__SN__] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(len(tiles),self.__SN__))
                                data[a:b,:] = self.O[:,:]
                            self.write_attrs(data)
                        if 'transitions' in features:
                            self.__buffer__ = 'transitions'
                            if not sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                                if self.__linear__:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__,
                                                                     (t*(self.__TN__**2),),dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__,
                                                                     (t*(self.__TN__**2),),dtype='f4',compression=self.__compression__,shuffle=True)
                                else:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__,
                                                                     (t,self.__TN__,self.__TN__),dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__,
                                                                     (t,self.__TN__,self.__TN__),dtype='f4',compression=self.__compression__,shuffle=True)
                            else: #need to delete the entry?
                                data = self.f[sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__]
                            self.O = mp.Array(ctypes.c_float,(self.__TN__**2)*len(tiles),lock=False)
                            for i in range(len(tiles)):                                              #make ||
                                core.transitions_bin(self.I,tiles[i][0],tiles[i][1]-1,                 #2D=> 1 lookback
                                                   self.B,self.O,self.__TN__,i*(self.__TN__**2))     #make ||
                            a,b = int(start/self.__window__),int(start/self.__window__+len(tiles))
                            if self.__linear__:
                                data[a*(self.__TN__**2):b*(self.__TN__**2)] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(len(tiles),self.__TN__,self.__TN__))
                                data[a:b,:,:] = self.O[:,:,:]
                            self.write_attrs(data)
        #TILES--------------------------------------------------------------------------------------------------TILES
        #FULL----------------------------------------------------------------------------------------------------FULL
        else:
            y = (end-start)-self.__window__
            for track in tracks:
                if track in self.A:
                    for rg in self.A[track]:
                        self.I = self.A[track][rg]
                        if 'moments' in features:
                            self.__buffer__ = 'moments'
                            if not sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                                if self.__linear__:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__)*self.__MN__,),
                                                                     dtype='f8',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__)*self.__MN__,),
                                                                     dtype='f8',compression=self.__compression__,shuffle=True)
                                else:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__),self.__MN__),
                                                                     dtype='f8',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__),self.__MN__),
                                                                     dtype='f8',compression=self.__compression__,shuffle=True)
                            else: #need to delete the entry?
                                data = self.f[sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/moments/%s'%self.__window__]
                            self.O = mp.Array(ctypes.c_double,self.__MN__*y,lock=False)
                            if not self.__fast__: #sliding approx algorithm
                                for i in range(y):
                                    core.exact_moments(self.I,i,i+self.__window__,self.O,i*self.__MN__)
                            else: #exact algorithm
                                core.sliding_moments(self.I,0,(end-start),self.__window__,self.__window__/self.__restart__,self.O)
                            if self.__linear__:
                                data[start*self.__MN__:(end-self.__window__)*self.__MN__] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(int(len(self.O)/self.__MN__),self.__MN__))
                                data[start:(end-self.__window__),:] = self.O[:,:]
                            self.write_attrs(data)
                        if 'spectrum' in features:
                            self.__buffer__ = 'spectrum'
                            if not sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                                if self.__linear__:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__)*self.__SN__,),
                                                                     dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                 ((self.__len__-self.__window__)*self.__SN__,),
                                                                 dtype='f4',compression=self.__compression__,shuffle=True)
                                else:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+list(seq.keys())[0]+'/'+track+'/spectrum/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__),self.__SN__),
                                                                     dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                 ((self.__len__-self.__window__)*self.__SN__,),
                                                                 dtype='f4',compression=self.__compression__,shuffle=True)
                            else: #need to delete the entry?
                                data = self.f[sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__]
                            self.O = mp.Array(ctypes.c_float,self.__SN__*y,lock=False)
                            core.sliding_spectrum_bin(self.I,0,(end-start),self.__window__,self.B,self.O)
                            if self.__linear__:
                                data[start*self.__SN__:(end-self.__window__)*self.__SN__] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(int(len(self.O)/self.__SN__),self.__SN__))
                                data[start:(end-self.__window__),:] = self.O[:,:]
                            self.write_attrs(data)
                        if 'transitions' in features:
                            self.__buffer__ = 'transitions'
                            if not sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                                if self.__linear__:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__)*self.__TN__**2,),
                                                                     dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                 ((self.__len__-self.__window__)*self.__SN__,),
                                                                 dtype='f4',compression=self.__compression__,shuffle=True)
                                else:
                                    if self.__compression__=='gzip':
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+list(seq.keys())[0]+'/'+track+'/transitions/%s'%self.__window__,
                                                                     ((self.__len__-self.__window__),self.__TN__,self.__TN__),
                                                                     dtype='f4',compression=self.__compression__,
                                                                     compression_opts=self.__ratio__,shuffle=True)
                                    else:
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/spectrum/%s'%self.__window__,
                                                                 ((self.__len__-self.__window__)*self.__SN__,),
                                                                 dtype='f4',compression=self.__compression__,shuffle=True)
                            else: #need to delete the entry?
                                data = self.f[sms[rg]+'/'+rg+'/'+self.__seq__+'/'+track+'/transitions/%s'%self.__window__]
                            self.O = mp.Array(ctypes.c_float,(self.__TN__**2)*(y-1),  lock=False)
                            core.sliding_transitions_bin(self.I,0,(end-start),self.__window__,self.B,self.O)
                            if self.__linear__:
                                data[start*(self.__TN__**2):(end-self.__window__-1)*(self.__TN__**2)] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(int(len(self.O)/(self.__TN__**2)),self.__TN__,self.__TN__))
                                data[start:(end-self.__window__-1),:,:] = self.O[:,:,:]
                            self.write_attrs(data)
        #FULL----------------------------------------------------------------------------------------------------FULL
        #POST----------------------------------------------------------------------------------------------------POST
        self.f.close()
        self.A = None
        self.I = None
        self.O = None
        if verbose: print('\nall writes completed to hdf5 container for %s-bp chunk'%(end-start))
        return True

    #|| on each seq into mp pool
    def extract_seq(self,alignment_path,base_name,sms,seq,
                    merge_rg=True,tracks=['total'],features=['moments'],verbose=False):
        k = list(seq.keys())[0]
        passes = int(k/self.__chunk__)
        last   = k%self.__chunk__
        if last == 0: last = []
        else:         last = [last]
        chunks = [self.__chunk__ for y in range(passes)]+last
        if verbose: print('seq=%s\ttracks=%s\tfeatures=%s'%(seq,','.join(tracks),','.join(features)))
        x = 0
        for i in range(len(chunks)):
            self.extract_chunk(alignment_path,base_name+'.seq.'+seq[k]+'.hdf5',sms,{seq[k]:k},x,x+chunks[i]+self.__window__, 
                               merge_rg=merge_rg,tracks=tracks,features=features,verbose=False)
            x += chunks[i]
        return True

    #we assume that the SAFE object has established the base windows via extract_seq and that
    #the user has defined valide tree parameters: self.___window_branch__ >= 2 and
    #self.__window_root__ >= self.__window_branch__*self.__window__ so that a valid root/stopping point
    #can be established:  data[sm][rg][seq][track][features][window]
    #workflow is that you make a new seqs folder and make new hdf5_windows that then get merged
    def update_seq_tree(self,hdf5_path,seq,verbose=False):
        k = list(seq.keys())[0]
        base = self.get_base_window_attributes(hdf5_path+'.seq.'+seq[k]+'.hdf5')  # get the base window attributes
        w,b,r = np.uint64(self.__window__),np.uint32(self.__window_branch__),min(np.uint64(self.__window_root__),np.uint64(k))
        if b>=2 and w*b<=r:
            self.f = File(hdf5_path+'.seq.'+seq[k]+'.hdf5','a')       #one input
            for sm in self.f:                                           #one sample
                if verbose: print('updating windows for sample %s'%sm)
                for rg in self.f[sm]:                                   #multiple read groups
                    if verbose: print('updating windows for read group %s'%rg)
                    if seq[k] in self.f[sm][rg]:
                        for track in self.f[sm][rg][seq[k]]:               #multiple tracks
                            if verbose: print('updating windows for track %s'%track)
                            if 'moments' in self.f[sm][rg][seq[k]][track]: #should have one windows size to start
                                if self.__tile__:
                                    if self.__linear__:
                                        w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                        data = self.f[sm][rg][seq[k]][track]['moments'][str(w)] #read from hdf5 once
                                        l = len(data)
                                        self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                        self.I[:] = data[:] #don't have to reshape to 1D array for linear
                                        self.O    = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False)
                                        while w*b<=r:
                                            if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t updating moments for %sbp to %sbp...'%(sm,rg,seq[k],track,w,w*b))
                                            w *= b
                                            l = int(l/b+(1 if l%b>0 else 0)) #update the new number of windows to use and window size
                                            core.merge_tiled_moments_target(self.I,self.O,b,disjoint=True)
                                            if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t completed %sbp to %sbp updates'%(sm,rg,seq[k],track,w,w*b))
                                            if self.__compression__=='gzip':
                                                if verbose: print('trying to create a new group for window %s'%w)
                                                data = self.f.create_dataset(sm+'/'+rg+'/'+seq[k]+'/'+track+'/moments/%s'%w,
                                                                             (l*self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,
                                                                             compression_opts=self.__ratio__,shuffle=True)
                                            else:
                                                data = self.f.create_dataset(sm+'/'+rg+'/'+seq[k]+'/'+track+'/moments/%s'%w,
                                                                             (l*self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,shuffle=True)
                                            data[:] = self.O[:]                                            # write it to hdf5
                                            self.__window__ = w                                            # update window size
                                            self.write_attrs(data)                                         # save attributes
                                            #reset data arrays---------------------------------------------------------------------------
                                            self.I     = mp.Array(ctypes.c_double,self.__MN__*l,lock=False)     #reset sratch
                                            self.I[:]  = self.O[:]                                              #copy back the last answer
                                            self.O     = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False) #reset output
                                    else:
                                        w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                        data = self.f[sm][rg][seq[k]][track]['moments'][str(w)] #read from hdf5 once
                                        l = len(data)
                                        self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                        self.I[:] = np.reshape(data,(self.__MN__*l,))[:] #reshape to 1D array for linear
                                        self.O    = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False)
                                        while w*b<=r:
                                            if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t updating moments for %sbp to %sbp...'%(sm,rg,seq[k],track,w,w*b))
                                            w *= b
                                            l = int(l/b+(1 if l%b>0 else 0)) #update the new number of windows to use and window size
                                            core.merge_tiled_moments_target(self.I,self.O,b,disjoint=True)
                                            if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t completed %sbp to %sbp updates'%(sm,rg,seq,track,w,w*b))
                                            if self.__compression__=='gzip':
                                                data = self.f.create_dataset(sm+'/'+rg+'/'+seq[k]+'/'+track+'/moments/%s'%w,
                                                                             (l,self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,
                                                                             compression_opts=self.__ratio__,shuffle=True)
                                            else:
                                                data = self.f.create_dataset(sm+'/'+rg+'/'+seq[k]+'/'+track+'/moments/%s'%w,
                                                                             (l,self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,shuffle=True)
                                            data[:] = np.reshape(self.O[:],(l,self.__MN__))[:]                  # write it to hdf5
                                            self.__window__ = w                                                 # update window size
                                            self.write_attrs(data)                                         # save attributes
                                            #reset data arrays---------------------------------------------------------------------------
                                            self.I     = mp.Array(ctypes.c_double,self.__MN__*l,lock=False)     #reset sratch
                                            self.I[:]  = self.O[:]                                              #copy back the last answer
                                            self.O     = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False) #reset output
                            self.set_attributes(base)  # must reset the self.__window__
                            if 'spectrum' in self.f[sm][rg][seq[k]][track]:
                                print('spectrum updates not implemented in HFM yet...')
                            if 'transitions' in self.f[sm][rg][seq[k]][track]:
                                print('transition updates not implemented in HFM yet...')
            self.f.close()
        return True

    #given an arbitrary hfm file, get and return the base attributes (IE the smallest windows)
    #we assume that the attributes for the base window summaries are uniform across the dimensions
    def get_base_window_attributes(self,hdf5_path):
        A = None
        if os.path.exists(hdf5_path):
            f = File(hdf5_path,'r')
            if len(f.keys())>0:
                sm = list(f.keys())[0]
                if len(f[sm].keys())>0:
                    rg = list(f[sm].keys())[0]
                    if len(f[sm][rg].keys())>0:
                        seq = list(f[sm][rg].keys())[0]
                        if len(f[sm][rg][seq].keys())>0:
                            trk = list(f[sm][rg][seq].keys())[0]
                            if len(f[sm][rg][seq][trk].keys())>0:
                                ftr = list(f[sm][rg][seq][trk].keys())[0]
                                if len(f[sm][rg][seq][trk][ftr].keys())>0:
                                    w = sorted([int(x) for x in list(f[sm][rg][seq][trk][ftr].keys())])[0]
                                    A = dict(f[sm][rg][seq][trk][ftr][str(w)].attrs)
            f.close()
        return A

    #given an arbitrary hfm file, get and return a seqname:seqlength dictionary
    def get_seqs(self,hdf5_path):
        seqs = {}
        if os.path.exists(hdf5_path):
            f = File(hdf5_path,'r')
            for sm in f:
                for rg in f[sm]:
                    for seq in f[sm][rg]:
                        seqs[str(seq)] = 0
                        for trk in f[sm][rg][seq]:
                            for ftr in f[sm][rg][seq][trk]:
                                w = sorted([int(x) for x in f[sm][rg][seq][trk][ftr].keys()])[0]
                                l = f[sm][rg][seq][trk][ftr][str(w)].attrs['len']
                                seqs[seq] = l
                                break
                            if seqs[seq]>0: break
            f.close()
        return seqs

    #we assume that the SAFE object has establish the base windows via extract_seq and that
    #the user may or may not have then used update_seq_tree to merge together wind_branch disjoint adjacent windows
    #we build a new hdf5 file by copying the base windows and then perform the update operations based on
    #the new given branch factor used when the HFM object was initialized in the form: sms/rgs/seqs/trks/frs/ws
    def rebranch_update_tree(self,hdf5_in_path,hdf5_out_path,window_branch=10,verbose=False):
        if os.path.exists(hdf5_in_path):
            base = self.get_base_window_attributes(hdf5_in_path) #get the base window attributes
            base['window_branch'] = window_branch                #this is the new branching factor update value
            seqs = self.get_seqs(hdf5_in_path)                   #gets the seqname:seqlength dict
            self.f = File(hdf5_in_path,'r')                      #opens the input hfm file
            self.g = File(hdf5_out_path,'a')                     #opens now in write mode
            self.set_attributes(base)                            #sets the object to have base window attributes of the file
            for seq in seqs:
                w,b,r = np.uint64(self.__window__),np.uint32(self.__window_branch__),min(np.uint64(self.__window_root__),np.uint64(seqs[seq]))
                if b>=2 and w*b<=r:
                    for sm in self.f:                                           #one sample
                        if verbose: print('updating windows for sample %s'%sm)
                        for rg in self.f[sm]:                                   #multiple read groups
                            if verbose: print('updating windows for read group %s'%rg)
                            if seq in self.f[sm][rg]:
                                for trk in self.f[sm][rg][seq]:               #multiple tracks
                                    if verbose: print('updating windows for track %s'%trk)
                                    if 'moments' in self.f[sm][rg][seq][trk]: #should have one windows size to start
                                        if self.__tile__:
                                            if self.__linear__:
                                                w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                                in_data = self.f[sm][rg][seq][trk]['moments'][str(w)] #read from hdf5 once
                                                l = len(in_data)
                                                #copies the original window size data from f to g---------------------------
                                                if self.__compression__=='gzip':
                                                    if verbose: print('trying to create a new group for window %s'%w)
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l*self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,
                                                                                 compression_opts=self.__ratio__,shuffle=True)
                                                else:
                                                    if verbose: print('trying to create a new group for window %s'%w)
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l*self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,shuffle=True)
                                                data[:] = in_data[:]   #copy all the data from one hfm to the other?
                                                self.write_attrs(data) #save attributes------------------------
                                                #copies the original window size data from f to g---------------------------
                                                self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                                self.I[:] = data[:] #don't have to reshape to 1D array for linear
                                                self.O    = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False)
                                                while w*b<=r:
                                                    if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t updating moments for %sbp to %sbp...'%(sm,rg,seq,trk,w,w*b))
                                                    w *= b
                                                    l = int(l/b+(1 if l%b>0 else 0)) #update the new number of windows to use and window size
                                                    core.merge_tiled_moments_target(self.I,self.O,b,disjoint=True)
                                                    if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t completed %sbp to %sbp updates'%(sm,rg,seq,trk,w,w*b))
                                                    if self.__compression__=='gzip':
                                                        if verbose: print('trying to create a new group for window %s'%w)
                                                        data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                     (l*self.__MN__,),dtype='f8',
                                                                                     compression=self.__compression__,
                                                                                     compression_opts=self.__ratio__,shuffle=True)
                                                    else:
                                                        if verbose: print('trying to create a new group for window %s'%w)
                                                        data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                     (l*self.__MN__,),dtype='f8',
                                                                                     compression=self.__compression__,shuffle=True)
                                                    data[:] = np.reshape(self.O[:],(l,self.__MN__))[:]                  # write it to hdf5
                                                    self.__window__ = w                                                 # update window size
                                                    self.write_attrs(data)                                         # save attributes
                                                    #reset data arrays---------------------------------------------------------------------------
                                                    self.I     = mp.Array(ctypes.c_double,self.__MN__*l,lock=False)     #reset sratch
                                                    self.I[:]  = self.O[:]                                              #copy back the last answer
                                                    self.O     = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False) #reset output
                                            else:
                                                w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                                in_data = self.f[sm][rg][seq][trk]['moments'][str(w)] #read from hdf5 once
                                                l = len(in_data)
                                                #------------------------------------------------------------------------------
                                                if self.__compression__=='gzip':
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l,self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,
                                                                                 compression_opts=self.__ratio__,shuffle=True)
                                                else:
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l,self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,shuffle=True)
                                                data[:] = in_data[:]   #copy all the data from one hfm to the other?
                                                self.write_attrs(data) #save attributes------------------------
                                                #------------------------------------------------------------------------------
                                                self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                                self.I[:] = np.reshape(data,(self.__MN__*l,))[:] #reshape to 1D array for linear
                                                self.O    = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False)
                                                while w*b<=r:
                                                    if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t updating moments for %sbp to %sbp...'%(sm,rg,seq,trk,w,w*b))
                                                    w *= b
                                                    l = int(l/b+(1 if l%b>0 else 0)) #update the new number of windows to use and window size
                                                    core.merge_tiled_moments_target(self.I,self.O,b,disjoint=True)
                                                    if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t completed %sbp to %sbp updates'%(sm,rg,seq,trk,w,w*b))
                                                    if self.__compression__=='gzip':
                                                        data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                     (l,self.__MN__,),dtype='f8',
                                                                                     compression=self.__compression__,
                                                                                     compression_opts=self.__ratio__,shuffle=True)
                                                    else:
                                                        data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                     (l,self.__MN__,),dtype='f8',
                                                                                     compression=self.__compression__,shuffle=True)
                                                    data[:] = np.reshape(self.O[:],(l,self.__MN__))[:]                  # write it to hdf5
                                                    self.__window__ = w                                                 # update window size
                                                    self.write_attrs(data)                                         # save attributes
                                                    #reset data arrays---------------------------------------------------------------------------
                                                    self.I     = mp.Array(ctypes.c_double,self.__MN__*l,lock=False)     #reset sratch
                                                    self.I[:]  = self.O[:]                                              #copy back the last answer
                                                    self.O     = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False) #reset output
                                    self.set_attributes(base) #must reset the self.__window__
                                    if 'spectrum' in self.f[sm][rg][seq][trk]:
                                        print('spectrum updates not implemented in HFM yet...')
                                    if 'transitions' in self.f[sm][rg][seq][trk]:
                                        print('transition updates not implemented in HFM yet...')
            self.f.close()
            self.g.close()
        return True

    # we assume that the SAFE object has establish the base windows via extract_seq and that
    # the user may or may not have then used update_seq_tree to merge together wind_branch disjoint adjacent windows
    # we build a new hdf5 file by copying the base windows and then perform the update operations based on
    # the new given branch factor used when the HFM object was initialized in the form: sms/rgs/seqs/trks/frs/ws
    def rebranch_update_seq_tree(self,hdf5_in_path,hdf5_out_path,seq,window_branch=10,comp='gzip',verbose=False):
        if os.path.exists(hdf5_in_path):
            base = self.get_base_window_attributes(hdf5_in_path) #get the base window attributes
            base['window_branch'] = window_branch                #this is the new branching factor update value
            self.__compression__ = comp
            self.f = File(hdf5_in_path,'r')                      #opens the input hfm file
            self.g = File(hdf5_out_path,'a')                     #opens now in write mode
            self.set_attributes(base)                            #sets the object to have base window attributes of the file
            seqs = {seq[list(seq.keys())[0]]:list(seq.keys())[0]}
            seq  = seq[list(seq.keys())[0]]
            w,b,r = np.uint64(self.__window__),np.uint32(self.__window_branch__),min(np.uint64(self.__window_root__),np.uint64(seqs[seq]))
            if b>=2 and w*b<=r:
                for sm in self.f:                                           #one sample
                    if verbose: print('updating windows for sample %s'%sm)
                    for rg in self.f[sm]:                                   #multiple read groups
                        if verbose: print('updating windows for read group %s'%rg)
                        if seq in self.f[sm][rg]:
                            for trk in self.f[sm][rg][seq]:               #multiple tracks
                                if verbose: print('updating windows for track %s'%trk)
                                if 'moments' in self.f[sm][rg][seq][trk]: #should have one windows size to start
                                    if self.__tile__:
                                        if self.__linear__:
                                            w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                            in_data = self.f[sm][rg][seq][trk]['moments'][str(w)] #read from hdf5 once
                                            l = len(in_data)/self.__MN__
                                            #------------------------------------------------------------------------------
                                            if self.__compression__=='gzip':
                                                if verbose: print('trying to create a new group for window %s'%w)
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l*self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,
                                                                             compression_opts=self.__ratio__,shuffle=True)
                                            else:
                                                if verbose: print('trying to create a new group for window %s'%w)
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l*self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,shuffle=True)
                                            data[:] = in_data[:]   #copy all the data from one hfm to the other?
                                            self.write_attrs(data) #save attributes------------------------
                                            #------------------------------------------------------------------------------
                                            self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                            self.I[:] = data[:] #don't have to reshape to 1D array for linear
                                            self.O    = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False)
                                            while w*b<=r:
                                                if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t updating moments for %sbp to %sbp...'%(sm,rg,seq,trk,w,w*b))
                                                w *= b
                                                l = int(l/b+(1 if l%b>0 else 0)) #update the new number of windows to use and window size
                                                core.merge_tiled_moments_target(self.I,self.O,b,disjoint=True)
                                                if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t completed %sbp to %sbp updates'%(sm,rg,seq,trk,w,w*b))
                                                if self.__compression__=='gzip':
                                                    if verbose: print('trying to create a new group for window %s'%w)
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l*self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,
                                                                                 compression_opts=self.__ratio__,shuffle=True)
                                                else:
                                                    if verbose: print('trying to create a new group for window %s'%w)
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l*self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,shuffle=True)
                                                data[:] = self.O[:]                                            # write it to hdf5
                                                self.__window__ = w                                            # update window size
                                                self.write_attrs(data)                                         # save attributes
                                                #reset data arrays---------------------------------------------------------------------------
                                                self.I     = mp.Array(ctypes.c_double,self.__MN__*l,lock=False)     #reset sratch
                                                self.I[:]  = self.O[:]                                              #copy back the last answer
                                                self.O     = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False) #reset output
                                        else:
                                            w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                            in_data = self.f[sm][rg][seq][trk]['moments'][str(w)] #read from hdf5 once
                                            l = len(in_data)
                                            #------------------------------------------------------------------------------
                                            if self.__compression__=='gzip':
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l,self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,
                                                                             compression_opts=self.__ratio__,shuffle=True)
                                            else:
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l,self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,shuffle=True)
                                            data[:] = in_data[:]   #copy all the data from one hfm to the other?
                                            self.write_attrs(data) #save attributes------------------------
                                            #------------------------------------------------------------------------------
                                            self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                            self.I[:] = np.reshape(data,(self.__MN__*l,))[:] #reshape to 1D array for linear
                                            self.O    = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False)
                                            while w*b<=r:
                                                if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t updating moments for %sbp to %sbp...'%(sm,rg,seq,trk,w,w*b))
                                                w *= b
                                                l = int(l/b+(1 if l%b>0 else 0)) #update the new number of windows to use and window size
                                                core.merge_tiled_moments_target(self.I,self.O,b,disjoint=True)
                                                if verbose: print('sample:%s\trg:%s\tseq:%s\ttrack:%s\t completed %sbp to %sbp updates'%(sm,rg,seq,trk,w,w*b))
                                                if self.__compression__=='gzip':
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l,self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,
                                                                                 compression_opts=self.__ratio__,shuffle=True)
                                                else:
                                                    data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                                 (l,self.__MN__,),dtype='f8',
                                                                                 compression=self.__compression__,shuffle=True)
                                                data[:] = np.reshape(self.O[:],(l,self.__MN__))[:]                  # write it to hdf5
                                                self.__window__ = w                                                 # update window size
                                                self.write_attrs(data)                                         # save attributes
                                                #reset data arrays---------------------------------------------------------------------------
                                                self.I     = mp.Array(ctypes.c_double,self.__MN__*l,lock=False)     #reset sratch
                                                self.I[:]  = self.O[:]                                              #copy back the last answer
                                                self.O     = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False) #reset output
                                    else:
                                        print('sliding window moment updates not implemented in HFM yet...')
                                self.set_attributes(base) #must reset the self.__window__
                                if 'spectrum' in self.f[sm][rg][seq][trk]:
                                    print('spectrum updates not implemented in HFM yet...')
                                if 'transitions' in self.f[sm][rg][seq][trk]:
                                    print('transition updates not implemented in HFM yet...')
            self.f.close()
            self.g.close()
        return True

    def clean_end_seq(self,hdf5_in_path,hdf5_out_path,seq,comp='gzip',max_correct=10,verbose=False):
        if os.path.exists(hdf5_in_path):
            base = self.get_base_window_attributes(hdf5_in_path) #get the base window attributes
            self.__compression__ = comp
            self.f = File(hdf5_in_path,'r')                      #opens the input hfm file
            self.g = File(hdf5_out_path,'a')                     #opens now in write mode
            self.set_attributes(base)                            #sets the object to have base window attributes of the file
            seqs = {seq[list(seq.keys())[0]]:list(seq.keys())[0]}
            seq  = seq[list(seq.keys())[0]]
            w,r = np.uint64(self.__window__),min(np.uint64(self.__window_root__),np.uint64(seqs[seq]))
            for sm in self.f:                                           #one sample
                    if verbose: print('cleaning windows for sample %s'%sm)
                    for rg in self.f[sm]:                                   #multiple read groups
                        if verbose: print('cleaning windows for read group %s'%rg)
                        if seq in self.f[sm][rg]:
                            for trk in self.f[sm][rg][seq]:               #multiple tracks
                                if verbose: print('cleaning windows for track %s'%trk)
                                if 'moments' in self.f[sm][rg][seq][trk]: #should have one windows size to start
                                    if self.__tile__:
                                        if self.__linear__:
                                            w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                            in_data = self.f[sm][rg][seq][trk]['moments'][str(w)] #read from hdf5 once
                                            l = len(in_data)/self.__MN__
                                            #------------------------------------------------------------------------------
                                            if self.__compression__=='gzip':
                                                if verbose: print('trying to create a new group for window %s'%w)
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l*self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,
                                                                             compression_opts=self.__ratio__,shuffle=True)
                                            else:
                                                if verbose: print('trying to create a new group for window %s'%w)
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l*self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,shuffle=True)
                                            #clean the ends of the data
                                            seq_len   = in_data.attrs['len']
                                            last_w    = 1.0*seq_len%w
                                            self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                            self.I[:] = in_data[:] #don't have to reshape to 1D array for linear
                                            i,x = (l-1)*self.__MN__,0
                                            if self.I[i]<last_w:
                                                if verbose: print('corrected last window size from %s to %s'%(self.I[i],last_w))
                                                self.I[i:i+self.__MN__] = [last_w,0.0,0.0,0.0,0.0,0.0,0.0,-3.0]
                                                i -= self.__MN__
                                                x += 1
                                            while(self.I[i]<w and x<max_correct):
                                                if verbose: print('corrected window index %s to size %s'%(i,w))
                                                self.I[i:i+self.__MN__] = [w,0.0,0.0,0.0,0.0,0.0,0.0,-3.0]
                                                i -= self.__MN__
                                                x += 1
                                            if verbose: print('corrected %s windows'%x)
                                            data[:] = self.I[:]
                                            self.write_attrs(data)  # save attributes------------------------
                                        else:
                                            w,b = np.uint64(self.__window__),np.uint32(self.__window_branch__)
                                            in_data = self.f[sm][rg][seq][trk]['moments'][str(w)] #read from hdf5 once
                                            l = len(in_data)
                                            #------------------------------------------------------------------------------
                                            if self.__compression__=='gzip':
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l,self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,
                                                                             compression_opts=self.__ratio__,shuffle=True)
                                            else:
                                                data = self.g.create_dataset(sm+'/'+rg+'/'+seq+'/'+trk+'/moments/%s'%w,
                                                                             (l,self.__MN__,),dtype='f8',
                                                                             compression=self.__compression__,shuffle=True)
                                            seq_len   = in_data.attrs['len']
                                            last_w    = 1.0*seq_len%w
                                            self.I    = mp.Array(ctypes.c_double,self.__MN__*l,lock=False) #set sratch
                                            self.I[:] = np.reshape(in_data,(self.__MN__*l,))[:] #reshape to 1D array for linear
                                            i,x = (l-1)*self.__MN__,0
                                            if self.I[i]<last_w:
                                                if verbose: print('corrected last window size from %s to %s'%(self.I[i],last_w))
                                                self.I[i:i+self.__MN__] = [last_w,0.0,0.0,0.0,0.0,0.0,0.0,-3.0]
                                                i -= self.__MN__
                                                x += 1
                                            while(self.I[i]<w and x<max_correct):
                                                if verbose: print('corrected window index %s to size %s'%(i,w))
                                                self.I[i:i+self.__MN__] = [w,0.0,0.0,0.0,0.0,0.0,0.0,-3.0]
                                                i -= self.__MN__
                                                x += 1
                                            if verbose: print('corrected %s windows'%x)
                                            data[:] = np.reshape(self.I[:],(l,self.__MN__))[:]
                                            self.write_attrs(data)  # save attributes------------------------
                                    else:
                                        print('sliding window moment cleaning not implemented in HFM ye')
                                if 'spectrum' in self.f[sm][rg][seq][trk]:
                                    print('spectrum cleaning not implemented in HFM yet...')
                                if 'transitions' in self.f[sm][rg][seq][trk]:
                                    print('transition cleaning not implemented in HFM yet...')
            self.f.close()
            self.g.close()

    def check(self):
        if self.__buffer__ == 'moments':
            if self.__linear__:
                return all([self.O[i*self.__MN__]==self.__window__ \
                            for i in range(int(len(self.O)/self.__MN__))])
            else:
                return all([self.O[i,self.__N__]==self.__window__ for i in range(self.O.shape[0])])
        if self.__buffer__ == 'spectrum':
            if self.__linear__:
                return all([np.sum(self.O[i*self.__SN__:(i+1)*self.__SN__])==self.__window__ \
                            for i in range(int(len(self.O)/self.__SN__))])
            else:
                return all([np.sum(self.O[i,:])==self.__window__ for i in range(len(self.O))])
        if self.__buffer__ == 'transitions':
            if self.__linear__:
                if not self.__tile__:
                    return all([np.sum(self.O[i*self.__TN__**2:(i+1)*self.__TN__**2])==self.__window__-1 \
                                for i in range(int((len(self.O)-1)/self.__TN__**2))])                          #2D is 1 lookback
                else:
                    return all([np.sum(self.O[i*self.__TN__**2:(i+1)*self.__TN__**2])==self.__window__-1 \
                                for i in range(int(len(self.O)/self.__TN__**2))])
            else:
                if not self.__tile__:
                    return all([np.sum(self.O[i,:,:])==self.__window__-1 for i in range(len(self.O)-1)]) #2D is 1 lookback
                else:
                    return all([np.sum(self.O[i,:,:])==self.__window__-1 for i in range(len(self.O))])   #2D is 1 lookback
            
    #given a buffered chunk, reform fro 1D to N    
    def reshape_buffer(self,feature='moments'):
        if   'moments' == feature:
            self.O = np.reshape(self.O,(int(len(self.O)/self.__MN__),self.__MN__))
        elif 'spectrum' == feature:
            self.O = np.reshape(self.O,(int(len(self.O)/self.__SN__),self.__SN__))
        elif 'transitions' == feature:
            self.O = np.reshape(self.O,(int(len(self.O)/(self.__TN__**2)),self.__TN__,self.__TN__))
        return True
    
    #save tranformations of a chunk
    def store_chunk(self, hdf5_path, sm, rg, seq,):
        return []

    #:::TO DO::: TRANSFORMATION COULD BE DONE IN CORE, OR MANY CAN BE DONE ON CLIENT/APPLICATION-SIDE

    #-------------------------------------------------------------------------------------------------------------------
    #returns buffer data attached to self.__buffer__[sm][rg][seq][trk][ftr][w]
    def buffer_chunk(self,hdf5_path,seq,start,end,
                     sms='all',rgs='all',tracks='all',features='all',windows='all',
                     verbose=False):
        C = {}
        if os.path.exists(hdf5_path):
            self.f = File(hdf5_path, 'r')
            if sms == 'all': sms = list(self.f.keys())
            for sm in sms:
                if sm in self.f.keys():
                    C[sm] = {}
                    if rgs == 'all': rgs = list(self.f[sm].keys())
                    if verbose: print(rgs)
                    for rg in rgs:
                        if rg in self.f[sm]:
                            C[sm][rg] = {}
                            if seq in self.f[sm][rg]:
                                if verbose: print(seq)
                                C[sm][rg][seq] = {}
                                if tracks == 'all': tracks = list(self.f[sm][rg][seq].keys())
                                if verbose: print(tracks)
                                for track in tracks:
                                    if verbose: print(track)
                                    if track in self.f[sm][rg][seq]:
                                        C[sm][rg][seq][track] = {}
                                        if features == 'all': features = list(self.f[sm][rg][seq][track].keys())
                                        if verbose: print(features)
                                        for feature in features:
                                            if feature == 'moments' and feature in self.f[sm][rg][seq][track]:
                                                C[sm][rg][seq][track][feature] = {}
                                                if windows == 'all': windows = list(self.f[sm][rg][seq][track][feature].keys())
                                                if verbose: print(windows)
                                                for w in windows:
                                                    if w in self.f[sm][rg][seq][track][feature]:
                                                        data = self.f[sm+'/'+rg+'/'+seq+'/'+track+'/moments/%s'%w]
                                                        self.read_attrs(data)
                                                        end = int(min(end,self.__len__)) #correct any loads that are out of bounds
                                                        if self.__tile__:
                                                            #[1] set the chunk tiles including the last partial
                                                            tiles = [[int(i),int(i+self.__window__)] for i in range(0,end-start,self.__window__)] #full windows
                                                            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
                                                            #[2] get the total container size, IE number of full (and partial) tiles
                                                            t = int(self.__len__/self.__window__)
                                                            if  self.__len__%self.__window__>0: t += 1

                                                            a,b = int(start/self.__window__),int(start/self.__window__+len(tiles))
                                                            if self.__linear__:
                                                                C[sm][rg][seq][track][feature][w]    = mp.Array(ctypes.c_double,self.__MN__*len(tiles),lock=False)
                                                                C[sm][rg][seq][track][feature][w][:] = data[a*self.__MN__:b*self.__MN__][:]
                                                            else:
                                                                C[sm][rg][seq][track][feature][w]      = np.zeros((len(tiles),self.__MN__),dtype='f8')
                                                                C[sm][rg][seq][track][feature][w][:,:] = data[a:b,:]
                                                        else:
                                                            y = int((end-start)-self.__window__)
                                                            if self.__linear__:
                                                                C[sm][rg][seq][track][feature][w]    = mp.Array(ctypes.c_double,self.__MN__*y,lock=False)
                                                                C[sm][rg][seq][track][feature][w][:] = data[start*self.__MN__:(end-self.__window__)*self.__MN__][:]
                                                            else:
                                                                C[sm][rg][seq][track][feature][w]      = np.zeros((y,self.__MN__),dtype='f8')
                                                                C[sm][rg][seq][track][feature][w][:,:] = data[start:(end-self.__window__),:]
                                            if feature == 'spectrum' and feature in self.f[sm][rg][seq][track]:
                                                C[sm][rg][seq][track][feature] = {}
                                                if windows == 'all': windows = list(self.f[sm][rg][seq][track][feature].keys())
                                                for w in windows:
                                                    if w in self.f[sm][rg][seq][track][feature]:
                                                        data = self.f[sm+'/'+rg+'/'+seq+'/'+track+'/spectrum/%s'%w]
                                                        self.read_attrs(data)
                                                        end = int(min(end,self.__len__))#correct any loads that are out of bounds
                                                        if self.__tile__:
                                                            #[1] set the chunk tiles including the last partial
                                                            tiles = [[int(i),int(i+self.__window__)] for i in range(0,end-start,self.__window__)] #full windows
                                                            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
                                                            #[2] get the total container size, IE number of full (and partial) tiles
                                                            t = int(self.__len__/self.__window__)
                                                            if  self.__len__%self.__window__>0: t += 1

                                                            a,b = int(start/self.__window__),int(start/self.__window__+len(tiles))
                                                            if self.__linear__:
                                                                C[sm][rg][seq][track][feature][w]    = mp.Array(ctypes.c_float,self.__SN__*len(tiles),lock=False)
                                                                C[sm][rg][seq][track][feature][w][:] = data[a*self.__SN__:b*self.__SN__][:]
                                                            else:
                                                                C[sm][rg][seq][track][feature][w]      = np.zeros((len(tiles),self.__SN__),dtype='f4')
                                                                C[sm][rg][seq][track][feature][w][:,:] = data[a:b,:]
                                                        else:
                                                            y = int((end-start)-self.__window__)
                                                            if self.__linear__:
                                                                C[sm][rg][seq][track][feature][w]    = mp.Array(ctypes.c_float,self.__SN__*y,lock=False)
                                                                C[sm][rg][seq][track][feature][w][:] = data[start*self.__SN__:(end-self.__window__)*self.__SN__][:]
                                                            else:
                                                                C[sm][rg][seq][track][feature][w]      = np.zeros((y,self.__SN__),dtype='f4')
                                                                C[sm][rg][seq][track][feature][w][:,:] = data[start:(end-self.__window__),:]
                                            if feature == 'transitions' and feature in self.f[sm][rg][seq][track]:
                                                C[sm][rg][seq][track][feature] = {}
                                                if windows == 'all': windows = list(self.f[sm][rg][seq][track][feature].keys())
                                                for w in windows:
                                                    if w in self.f[sm][rg][seq][track][feature]:
                                                        data = self.f[sm+'/'+rg+'/'+seq+'/'+track+'/transitions/%s'%w]
                                                        self.read_attrs(data)
                                                        end = int(min(end,self.__len__)) #correct any loads that are out of bounds
                                                        if self.__tile__:
                                                            #[1] set the chunk tiles including the last partial
                                                            tiles = [[int(i),int(i+self.__window__)] for i in range(0,end-start,self.__window__)] #full windows
                                                            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
                                                            #[2] get the total container size, IE number of full (and partial) tiles
                                                            t = int(self.__len__/self.__window__)
                                                            if  self.__len__%self.__window__>0: t += 1

                                                            a,b = int(start/self.__window__),int(start/self.__window__+len(tiles))
                                                            if self.__linear__:
                                                                C[sm][rg][seq][track][feature][w]    = mp.Array(ctypes.c_float,(self.__TN__**2)*len(tiles),lock=False)
                                                                C[sm][rg][seq][track][feature][w][:] = data[a*(self.__TN__**2):b*(self.__TN__**2)][:]
                                                            else:
                                                                C[sm][rg][seq][track][feature][w]        = np.zeros((len(tiles),self.__TN__,self.__TN__),dtype='f4')
                                                                C[sm][rg][seq][track][feature][w][:,:,:] = data[a:b,:,:]
                                                        else:
                                                            y = int((end-start)-self.__window__)
                                                            if self.__linear__:
                                                                C[sm][rg][seq][track][feature][w]    = mp.Array(ctypes.c_float,(self.__TN__**2)*y,lock=False)
                                                                C[sm][rg][seq][track][feature][w][:] = data[start*(self.__TN__**2):(end-self.__window__)*(self.__TN__**2)][:]
                                                            else:
                                                                C[sm][rg][seq][track][feature][w]        = np.zeros((y,self.__TN__,self.__TN__),dtype='f4')
                                                                C[sm][rg][seq][track][feature][w][:,:,:] = data[start:(end-self.__window__),:,:]
            self.__buffer__ = C
            self.f.close()
        else:
            print('hdf5_path = %s was not found.'%hdf5_path)
            return False
        return True

    #slow performance should just use json.dumps(default=handler) instead...
    def buffer_chunk_json(self,hdf5_path,seq,start,end,
                          sms='all',rgs='all',tracks='all',features='all',windows='all',
                          verbose=False):
        self.buffer_chunk(hdf5_path,seq,start,end,sms=sms,rgs=rgs,tracks=tracks,features=features,verbose=verbose)
        for sm in self.__buffer__:
            for rg in self.__buffer__[sm]:
                for seq in self.__buffer__[sm][rg]:
                    for trk in self.__buffer__[sm][rg][seq]:
                        for ftr in self.__buffer__[sm][rg][seq][trk]:
                            for w in self.__buffer__[sm][rg][seq][trk][ftr]:
                                self.__buffer__[sm][rg][seq][trk][ftr][w] = self.__buffer__[sm][rg][seq][trk][ftr][w].tolist()
        return True

    #:::TO DO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def buffer_seq(self,hdf5_path,seq,tracks='all',features='all',windows='all',verbose=False):
        k = list(seq.keys())[0]
        passes = int(k/self.__chunk__)
        last   = k%self.__chunk__
        if last == 0: last = []
        else:         last = [last]
        chunks = [self.__chunk__ for y in range(passes)]+last
        x = 0
        for i in range(len(chunks)):
            self.buffer_chunk(hdf5_path, seq, x, x+chunks[i],
                              tracks=tracks,features=features,windows=windows,verbose=False)
            x += chunks[i]
        return True
    #-------------------------------------------------------------------------------------------------------------------
    #we assume that you have called buffer_chunk and now have data inside __buffer__
    #will take the largest window with l windows and collapse them into one MN size vector
    def chunk_collapse(self,verbose=False):
        if self.__tile__:
            if self.__linear__:
                for sm in self.__buffer__:
                    for rg in self.__buffer__[sm]:
                        for seq in self.__buffer__[sm][rg]:
                            for trk in self.__buffer__[sm][rg][seq]:
                                if 'moments' in self.__buffer__[sm][rg][seq][trk]:
                                    ftr = 'moments' #now take largest window from avaible
                                    w = sorted(list(self.__buffer__[sm][rg][seq][trk][ftr].keys()),key=lambda x: int(x))[0]
                                    data = self.__buffer__[sm][rg][seq][trk][ftr][w]
                                    l = len(data)
                                    self.I    = mp.Array(ctypes.c_double,int(self.__MN__*l),lock=False)
                                    self.I[:] = data[:] #don't have to reshape to 1D array for linear
                                    self.O    = mp.Array(ctypes.c_double,int(self.__MN__),lock=False)
                                    core.merge_tiled_moments_target(self.I,self.O,np.uint32(l),True)
                                    #self.O needs to be reshapped for np arrays right?
                                    #self.__len__ needs to be cast to str for keys...
                                    self.__buffer__[sm][rg][seq][trk][ftr][str(self.__len__)] = self.O #set a new window
            else:
                for sm in self.__buffer__:
                    for rg in self.__buffer__[sm]:
                        for seq in self.__buffer__[sm][rg]:
                            for trk in self.__buffer__[sm][rg][seq]:
                                if 'moments' in self.__buffer__[sm][rg][seq][trk]:
                                    ftr = 'moments' #now take largest window from avaible
                                    w = sorted(list(self.__buffer__[sm][rg][seq][trk][ftr].keys()),key=lambda x: int(x))[0]
                                    data = self.__buffer__[sm][rg][seq][trk][ftr][w]
                                    l = len(data)
                                    self.I    = mp.Array(ctypes.c_double,int(self.__MN__*l),lock=False)
                                    self.I[:] = np.reshape(data,(self.__MN__*l,))[:] #reshape for non-linear
                                    self.O    = mp.Array(ctypes.c_double,int(self.__MN__),lock=False)
                                    core.merge_tiled_moments_target(self.I,self.O,np.uint32(l*self.__MN__),True)
                                    self.__buffer__[sm][rg][seq][trk][ftr][str(self.__len__)] = self.O #set a new window
        else:
            #sliding
            #:::TO DO:::
            print('not implemented yet..')
        return True

    #given an hdf5 file, generate a dict of zero coordinates using the smallest windows...
    def get_maxima_regions(self,hdf5_paths,out_dir=None,lower_cut=0.1,upper_cut=1000.0,min_w=500,verbose=True):
        if out_dir is not None and os.path.exists(out_dir+'/sms.maxima.json'):
            with open(out_dir+'/sms.maxima.json','r') as f:
                I = json.load(f)
        else:
            R,I = {},{}
            if type(hdf5_paths) is not list: hdf5_paths = [hdf5_paths]
            for hdf5_path in hdf5_paths:
                if verbose: print('generating maxima ranges for %s'%hdf5_path.rsplit('/')[-1])
                A = get_hdf5_attrs(hdf5_path)
                for sm in A:
                    R[sm] = {}
                    for rg in A[sm]:
                        R[sm][rg] = {}
                        for seq in A[sm][rg]:
                            start = time.time()
                            R[sm][rg][seq] = []
                            w = int(sorted(list(A[sm][rg][seq].keys()),key=lambda x: int(x))[0])
                            l = A[sm][rg][seq][str(w)]['len']
                            m = A[sm][rg][seq][str(w)]['M1']
                            self.buffer_chunk(hdf5_path,seq=seq,start=0,end=l,
                                              sms=[sm],rgs=[rg],tracks=['total'],
                                              features=['moments'],windows=[str(w)])
                            data = self.__buffer__
                            mmts = data[sm][rg][seq]['total']['moments'][str(w)]
                            i = 0
                            while i < len(mmts):
                                j = i
                                while j < len(mmts) and (mmts[j][m]<lower_cut or mmts[j][m]>upper_cut): j += 1
                                if i<j: R[sm][rg][seq] += [[w*i,w*j]]
                                i = j+1
                            if len(R[sm][rg][seq])>0 and R[sm][rg][seq][-1][1]>l: R[sm][rg][seq][-1][1]=l
                            self.__buffer__ = None
                            stop = time.time()
                            if verbose: print('processed sm=%s rg=%s seq%s in %s secs'%(sm,rg,seq,round(stop-start,2)))
            sms = sorted(list(R.keys()))
            rg = list(R[sms[0]].keys())[0]
            I = {}
            for seq in R[sms[0]][rg]: I[seq] = R[sms[0]][rg][seq]
            for i in range(len(R)):
                for rg in R[sms[i]]:
                    for seq in R[sms[i]][rg]:
                        I[seq] = core.LRF_1D(I[seq],R[sms[i]][rg][seq])[0]
            if verbose: print('intersecting, extending and filtering ranges across samples:%s'%sorted(list(R.keys())))
            R = []
            for seq in I:
                T = []
                for i in range(len(I[seq])): #[1] filter ranges that are too small
                    if I[seq][i][1]-I[seq][i][0] >= min_w: T += [I[seq][i]]
                I[seq] = T
                for i in range(len(I[seq])-1):
                    if I[seq][i+1][0]-I[seq][i][1] <= min_w: I[seq][i][1] += min_w
                I[seq] = core.LRF_1D(I[seq],I[seq])[1] #union of the gap extended sections
                if verbose: print('%s intersected, extended, ranges passed filters for seq=%s'%(len(I[seq]),seq))
            if out_dir is not None:
                if not os.path.exists(out_dir): os.mkdir(out_dir)
                with open(out_dir+'/sms.maxima.json','w') as f:
                    json.dump(I,f)
        return I

    #given an hdf5 default file, standardizes M1,M2,M3,M4 and decorreletas total from GC to derive the RD signal
    #then writes smaller hdf5 file to hdf5 out file where coordinate based sampling can then occur
    #tracks are the final tracks selected, mask is a set of {seq:[start,end] regiond, coords is {seq:[start,end] regions
    #hsdf5_out_path will be file://sm/rg/label/cid
    def preprocess_tracks(self,hdf5_in_path,tracks,mask,coords,hdf5_out_path,
                          decorrelate=True,normalize=False,verbose=True):
        C = {}
        A = get_hdf5_attrs(hdf5_in_path)
        for sm in A:
            C[sm] = {}
            for rg in A[sm]:
                C[sm][rg] = {}
                for seq in A[sm][rg]:
                    start = time.time()
                    C[sm][rg][seq],data = {},{}
                    w = int(sorted(list(A[sm][rg][seq].keys()), key=lambda x: int(x))[0])
                    l = A[sm][rg][seq][str(w)]['len']
                    self.buffer_chunk(hdf5_in_path,seq,0,l,sms=[sm],rgs=[rg],features=['moments'],windows=[str(w)])
                    for k in self.__buffer__[sm][rg][seq]:
                        data[k] = core.standardized_moments(self.__buffer__[sm][rg][seq][k]['moments'][str(w)])
                    #remove the mask windows from the full set using LRF rev-difference function
                    neg,pos,idx = [list(x) for x in np.asarray(mask[seq])/w],[[0,l/w]],[]
                    for x in core.LRF_1D(neg,pos)[3]: idx += range(x[0],x[1]+1,1)
                    for k in data: data[k] = data[k][idx,:]
                    #cross-correlation filter---------------------------------------------------
                    kernel,kernel_function,track,bias = 16,'lin','primary','GC'
                    if u'RD' not in tracks: tracks += [u'RD']
                    data[u'RD'] = np.zeros(data[track].shape,dtype=np.float64)
                    #kernel weights----------------------------------------------
                    if kernel_function=='exp':   kernel_ws = np.exp(range(0,kernel)+[kernel]+range(0,kernel)[::-1])
                    elif kernel_function=='lin': kernel_ws = range(0,kernel)+[kernel]+range(0,kernel)[::-1]
                    elif kernel_function=='log': kernel_ws = np.log1p(range(0,kernel)+[kernel]+range(0,kernel)[::-1])
                    else:                        kernel_ws = [1 for x in range(2*kernel+1)]
                    min_k,max_k = np.min(kernel_ws),np.max(kernel_ws)
                    kernel_ws -= min_k
                    kernel_ws /= max_k
                    # kernel weights----------------------------------------------

                    for d in range(data[track].shape[1]):
                        sig  = data[track][:,d]/(1.0+data[bias][:,d])
                        min_s,max_s = np.min(sig),np.max(sig)
                        min_t,max_t = np.min(data[track][:,d]),np.max(data[track][:,d])
                        sig -= min_s
                        sig /= max_s-min_s
                        sig *= max_t-min_t
                        sig += min_t
                        for i in range(kernel,len(sig)-kernel-1,1):
                            data[u'RD'][i,d] = np.average(sig[i-kernel:i+kernel+1],weights=kernel_ws)

                    #signal processing filters--------------------------------------------------
                    data[u'WD'] = np.copy(data[u'primary'])
                    sig = np.copy(data[u'RD'])                                     # intial copy
                    mins,maxs = {},{}
                    for d in range(sig.shape[1]):
                        mins[d],maxs[d] = np.min(sig[:,d]),np.max(sig[:,d])        # scale into
                        sig[:,d] -= mins[d]                                        # [0,1]
                        sig[:,d] /= maxs[d]-mins[d]                                # range, then into
                    x = int(math.floor(math.pow(data['RD'].shape[0],0.5)))
                    sig = np.reshape(sig[:x*x],(x,x,4))
                    #filt = skimage.restoration.denoise_wavelet(sig,wavelet='haar')
                    filt = skimage.restoration.denoise_bilateral(sig,sigma_color=0.1,sigma_spatial=15)
                    # filt = skimage.restoration.denoise_tv_bregman(sig,weight=0.001,
                    #                                               max_iter=500,eps=0.001,
                    #                                               isotropic=True)
                    filt = np.reshape(filt,(x*x,4))
                    fmins,fmaxs = {},{}
                    for d in range(filt.shape[1]):
                        fmins[d],fmaxs[d] = np.min(filt[:,d]),np.max(filt[:,d])
                        filt[:,d] -= fmins[d]
                        filt[:,d] /= fmaxs[d]-fmins[d]
                        filt[:,d] *= maxs[d]-mins[d]
                        filt[:,d] += mins[d]
                    data[u'WD'][:x*x,:] = filt[:]

                    #cross-correlation filter---------------------------------------------------
                    # data[u'RD'] = self.track_decorrelation(data,track='primary',bias='GC',
                    #                                        kernel=2,kernel_function='exp',
                    #                                        lower_res=True,rescale=True)
                    # if decorrelate:

                    ks = []
                    for k in data:
                        if k in tracks: ks += [k]
                    ks = sorted(list(ks))

                    # if normalize: data = self.track_normalize(data,[0.0,1.0])

                    for i,j in it.combinations(range(len(ks)),2):
                        i,j = sorted([i,j])
                        C[sm][rg][seq][(i,j)] = np.zeros((data[ks[i]].shape[1],),dtype='f8')
                        for d in range(data[ks[i]].shape[1]):
                            C[sm][rg][seq][(i,j)][d] = np.corrcoef(data[ks[i]][:,d],data[ks[j]][:,d])[0][1]
                    stop = time.time()
                    if verbose: print('finished %s corr coeffs for seq=%s in %s sec'%\
                                      (len(C[sm][rg][seq]),seq,round(stop-start,2)))
        return C

    #finds, min and max and uses that to scale to 0.0 to 1.0 range
    def track_normalize(self,data,final=[0.0,1.0]):
        offset,scale = final[0],final[1]-final[0]
        ks = sorted(list(data.keys))
        for i in range(len(ks)):
            for d in range(data[ks[i]].shape[1]):
                min_v   = np.min(data[ks[i]][:,d])
                max_v   = np.max(data[ks[i]][:,d])
                range_v = max_v-min_v
                for j in range(data[ks[i]].shape[0]):
                    data[ks[i]][j,d] = ((data[ks[i]][j,d]-min_v)/range_v)*scale+offset
        return data

    #if correlation between track, can decorrelate such as GC bias from total/primary alignments
    def track_decorrelation(self,data,track='primary',bias='GC',kernel=32,kernel_function='exp',lower_res=True,rescale=False):
        D = np.zeros(data[track].shape,dtype=np.float64)
        for d in range(data[track].shape[1]):
            min_v  = np.min(data[bias][:,d])
            max_v  = np.max(data[bias][:,d])
            med_v  = np.median(data[bias][:,d])
            if max_v-min_v>0.0:
                if lower_res: #use 3/4 of bins below the median value
                    if med_v<=0.0: med_v = (max_v-min_v)/2.0
                    bins_v = list(np.arange(min_v,2*med_v,(2*med_v-min_v)/(1.0*(kernel/2)/2.0)))+\
                             list(np.arange(med_v+(med_v-min_v),max_v,max_v/(1.0*(kernel/2)/2.0)))+[max_v+1.0]
                else:
                    bins_v = np.histogram_bin_edges(data[bias][:,d],bins=kernel/2)
                hist_v = {b:[] for b in bins_v[:-1]}
                for i in range(len(data[bias][:,d])):
                    for b in range(len(bins_v)-1):
                        if data[bias][i,d]>=bins_v[b] and data[bias][i,d]<bins_v[b+1]:
                            hist_v[bins_v[b]] += [data[bias][i,d]]
                            break
                for b in range(len(bins_v)-1):
                    if len(hist_v[bins_v[b]])>0.0: hist_v[bins_v[b]] = np.mean(hist_v[bins_v[b]])
                    else:                          hist_v[bins_v[b]] = (bins_v[b]+bins_v[b+1])/2.0
                if kernel_function=='exp':   kernel_ws = np.exp(range(0,kernel)+[kernel]+range(0,kernel)[::-1])
                elif kernel_function=='lin': kernel_ws = range(0,kernel)+[kernel]+range(0,kernel)[::-1]
                elif kernel_function=='log': kernel_ws = np.log1p(range(0,kernel)+[kernel]+range(0,kernel)[::-1])
                else:                        kernel_ws = [1 for x in range(2*kernel+1)]
                min_k,max_k = np.min(kernel_ws),np.max(kernel_ws)
                kernel_ws -= min_k
                kernel_ws /= max_k
                for i in range(kernel,len(data[track])-kernel-1,1): #smoothing filter here with width>0
                    mu_rd_width = np.average(data[track][i-kernel:i+kernel+1,d],weights=kernel_ws)
                    for b in range(len(bins_v)-1):
                        if data[bias][i,d]>=bins_v[b] and data[bias][i,d]<bins_v[b+1]:
                            mu_bin_v = hist_v[bins_v[b]]
                            break
                    #D[i,d] = data[track][i,d]*(mu_rd/(mu_rd if mu_bin_v<=0.0 else mu_bin_v))
                    D[i,d] = np.average(data[track][i-kernel:i+kernel+1,d],weights=kernel_ws)*\
                             (mu_rd_width/(mu_rd_width if mu_bin_v==0.0 else mu_bin_v))
                if rescale:
                    min_t,min_b   = np.min(data[track][:,d]),np.min(D[:,d])
                    max_t,max_b   = np.max(data[track][:,d]),np.max(D[:,d])
                    D[:,d] -= min_b
                    D[:,d] /= max_b-min_b
                    D[:,d] *= max_t-min_t
                    D[:,d] += min_t
            else:
                D[:,d] = data[track][:,d]
        return D

    #feature access from the container here-----------------------
    def window_range(self,f,normalized=False):
        if not normalized:
            return f[self.__MAX__]-f[self.__MIN__]
        else:
            return (f[self.__MAX__]-f[self.__MIN__])/1.0*f[self.__N__]

    def window_var(self,f):
        if f[self.__N__]>1:
            return f[self.__M2__]/(f[self.__N__]-1)
        else:
            return 0.0
            
    def window_std(self,f):
        if f[self.__M2__]>0.0 and f[self.__N__]>1:
            return math.pow(f[self.__M2__]/(f[self.__N__]-1),0.5)
        else:
            return 0.0
            
    def window_skew(self,f):
        if f[self.__M2__] > 0.0:
            return math.pow(f[self.__N__],0.5)*f[self.__M3__]/math.pow(f[self.__M2__],1.5)
        else:
            return 0.0
            
    def window_kur(self,f):
        if f[self.__M2__] > 0.0:
            return f[self.__N__]*f[self.__M4__]/f[self.__M2__]**2 - 3.0
        else:
            return -3.0
            
    #get the ith spectrum as a 1D array
    def get_spectrum(self,S,B,i):
        b = len(B)-1
        X = np.zeros((b,),dtype='f4')
        X[:] = S[i*b:(i+1)*b]
        return  X      
    
    #get the ith transition matrix as a 2D array

    def get_transitions(self,T,B,i):
        b = len(B)-1
        bb = b*b
        X = np.zeros((b,b),dtype='f4')
        for j in range(b):
            X[j,:] = T[i*bb + j*b : i*bb + (j+1)*b]
        return X
    
    #pseudo-count and normalize spectrum and transitions
    
    #pretty print for transition matrices
    def print_transitions(self,T,B,i):
        b = len(B)-1
        bb = b*b
        X = np.zeros((b,b),dtype='f4')
        for j in range(b):
            X[j,:] = T[i*bb + j*b : i*bb + (j+1)*b]
        for j in X: print(j)
