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
import math
import glob
import numpy as np
import multiprocessing as mp
import pysam
import core

#alignemnt file meta-data utilties-------------------------------------------------------------------

#given an alignment file, SAM,BAM,CRAM reads the sequences and creates a name to length dict
#mapping the name and size of each sequence that will be passed to the extractor
#common workflows are to remove the sequences or match to the sequences that are desired
#before passing into a feature extraction runner
def get_seq(alignment_path,sort_by_largest=True):
    am = pysam.AlignmentFile(alignment_path,'rb')
    seqs = {s['SN']:s['LN'] for s in am.header['SQ']}
    am.close()
    return sorted([{seqs[k]:k} for k in seqs], key=lambda x: x, reverse=sort_by_largest)

#given an alignment file, SAM, BAM,CRAM reads the sequences and creates a dict mapping:
#{sample_name:[rg1_name,rg2_name,...rgx_name}
def get_rg(alignment_path):
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
def get_sm(alignment_path):
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
                 bins=None,tile=True,fast=True,linear=False,compression='lzf',ratio=9):
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
        self.__restart__ = 4                      #restart divisor of window if using fast algo
        self.__compression__ = compression        #can use lzf for fast float or gzip here
        self.__ratio__ = ratio                    #comp ratio 1-9
        if not bins is None:                      #bins used to generate counts
            self.__bins__ = bins                  #user definable
        else: #can add different bin functions here later...
            self.__bins__ = range(0,self.__max_depth__,self.__max_depth__/int(1E1))+[np.uint32(-1)]
        self.B = mp.Array(ctypes.c_float,self.__bins__,lock=False)
        self.__fast__ = fast #fast implies that you are doing a sliding window calculation
        #moment positions for easy future additions
        self.__N__,self.__SUM__,self.__MIN__,self.__MAX__, \
        self.__M1__,self.__M2__,self.__M3__,self.__M4__ = range(8)
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
        self.B = mp.Array(ctypes.c_float,self.__bins__,lock=False)
  
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
        self.__seq__ = seq.keys()[0]
        self.__len__ = seq[seq.keys()[0]]           
        self.ap = alignment_path
        if merge_rg: sms = {'all':'-'.join(sorted(list(set(sms.values()))))}
        if len(tracks)==1 and tracks[0]=='all':     tracks = self.__tracks__
        if len(features)==1 and features[0]=='all': features = self.__features__
        if verbose: print '|',
        end = min(end,seq[seq.keys()[0]]) #correct any loads that are out of bounds
        self.A = core.load_reads_all_tracks(self.ap,sms,self.__seq__,start,end,merge_rg) #dict, single loading
        self.f = File(hdf5_path, 'a')
        #PRE------------------------------------------------------------------------------------------------------PRE
        #TILES--------------------------------------------------------------------------------------------------TILES
        if self.__tile__:
            #[1] set the chunk tiles including the last partial
            tiles = [[i,i+self.__window__] for i in range(0,end-start,self.__window__)] #full windows
            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
            #[2] set the total container size in case it is not intialized
            t = self.__len__/self.__window__
            if  self.__len__%self.__window__>0: t += 1
            #[3] iterate on the tracks return from the input buffer tracks
            for track in tracks:
                if track in self.A:
                    for rg in self.A[track]:
                        self.I = self.A[track][rg] #pull out the track here-----------------------------------------------
                        if 'moments' in features:
                            self.__buffer__ = 'moments'
                            if not sms[rg]+'/'+rg+'/'+seq.keys()[0]+'/'+track+'/moments/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
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
                            if verbose: print '%s:M:%s'%(track,[1 if i else 0 for i in [self.check()]][0]),
                            a,b = (start/self.__window__),(start/self.__window__+len(tiles))
                            if self.__linear__:
                                data[a*self.__MN__:b*self.__MN__] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(len(self.O)/self.__MN__,self.__MN__))
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
                                
                            if verbose: print '%s:S:%s'%(track,[1 if i else 0 for i in [self.check()]][0]),
                            a,b = (start/self.__window__),(start/self.__window__+len(tiles))
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
                            if verbose: print '%s:T:%s'%(track,[1 if i else 0 for i in [self.check()]][0]),
                            a,b = (start/self.__window__),(start/self.__window__+len(tiles))
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
                            if verbose: print '%s:M:%s'%(track,[1 if i else 0 for i in [self.check()]][0]),
                            if self.__linear__:
                                data[start*self.__MN__:(end-self.__window__)*self.__MN__] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(len(self.O)/self.__MN__,self.__MN__))
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
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+seq.keys()[0]+'/'+track+'/spectrum/%s'%self.__window__,
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
                            if verbose: print '%s:S:%s'%(track,[1 if i else 0 for i in [self.check()]][0]),
                            if self.__linear__:
                                data[start*self.__SN__:(end-self.__window__)*self.__SN__] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(len(self.O)/self.__SN__,self.__SN__))
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
                                        data = self.f.create_dataset(sms[rg]+'/'+rg+'/'+seq.keys()[0]+'/'+track+'/transitions/%s'%self.__window__,
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
                            if verbose: print '%s:T:%s'%(track,[1 if i else 0 for i in [self.check()]][0]),
                            if self.__linear__:
                                data[start*(self.__TN__**2):(end-self.__window__-1)*(self.__TN__**2)] = self.O[:]
                            else:
                                self.O = np.reshape(self.O,(len(self.O)/(self.__TN__**2),self.__TN__,self.__TN__))
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
        k = seq.keys()[0]
        passes = k/self.__chunk__
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
        k = seq.keys()[0]
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
                                            core.merge_tiled_moments_target(self.I,self.O,np.b,disjoint=True)
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
                                            data[:] = np.reshape(self.O[:],(l,self.__MN__))[:]                  # write it to hdf5
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
                                            #reset data arrays---------------------------------------------------------------------------
                                            self.I     = mp.Array(ctypes.c_double,self.__MN__*l,lock=False)     #reset sratch
                                            self.I[:]  = self.O[:]                                              #copy back the last answer
                                            self.O     = mp.Array(ctypes.c_double,int(self.__MN__*(l/b+(1 if l%b>0 else 0))),lock=False) #reset output
            self.f.close()
        return True

    def check(self):
        if self.__buffer__ == 'moments':
            if self.__linear__:
                return all([self.O[i*self.__MN__]==self.__window__ \
                            for i in range(len(self.O)/self.__MN__)])
            else:
                return all([self.O[i,self.__N__]==self.__window__ for i in range(self.O.shape[0])])
        if self.__buffer__ == 'spectrum':
            if self.__linear__:
                return all([np.sum(self.O[i*self.__SN__:(i+1)*self.__SN__])==self.__window__ \
                            for i in range(len(self.O)/self.__SN__)])       
            else:
                return all([np.sum(self.O[i,:])==self.__window__ for i in range(len(self.O))])
        if self.__buffer__ == 'transitions':
            if self.__linear__:
                if not self.__tile__:
                    return all([np.sum(self.O[i*self.__TN__**2:(i+1)*self.__TN__**2])==self.__window__-1 \
                                for i in range((len(self.O)-1)/self.__TN__**2)])                          #2D is 1 lookback
                else:
                    return all([np.sum(self.O[i*self.__TN__**2:(i+1)*self.__TN__**2])==self.__window__-1 \
                                for i in range(len(self.O)/self.__TN__**2)]) 
            else:
                if not self.__tile__:
                    return all([np.sum(self.O[i,:,:])==self.__window__-1 for i in range(len(self.O)-1)]) #2D is 1 lookback
                else:
                    return all([np.sum(self.O[i,:,:])==self.__window__-1 for i in range(len(self.O))])   #2D is 1 lookback
            
    #given a buffered chunk, reform fro 1D to N    
    def reshape_buffer(self,feature='moments'):
        if   'moments' == feature:
            self.O = np.reshape(self.O,(len(self.O)/self.__MN__,self.__MN__))
        elif 'spectrum' == feature:
            self.O = np.reshape(self.O,(len(self.O)/self.__SN__,self.__SN__))
        elif 'transitions' == feature:
            self.O = np.reshape(self.O,(len(self.O)/(self.__TN__**2),self.__TN__,self.__TN__))
        return True
    
    #save tranformations of a chunk
    def store_chunk(self, hdf5_path, sm, rg, seq,):
        return []

    #:::TO DO::: TRANSFORMATION SHOULD BE DONE IN CORE...

    #-------------------------------------------------------------------------------------------------------------------
    #:::TO DO::: buffer methods can be rewritten to achieve a IGV stype API or other: sm,rg,seq,track,feature,start,stop
    #-------------------------------------------------------------------------------------------------------------------
    #select entries from the hdf5 store and buffer into memory
    #seq would match the self.__seq__ value and you could check
    #:::TO DO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def buffer_chunk(self, hdf5_path, sm, rg, seq, start, end,
                     tracks=['reads_all'],features=['moments'],verbose=False):
        #pull meta data for the chuck and set: window, bins, momentsm etc...
        if os.path.exists(hdf5_path):
            self.f = File(hdf5_path, 'r')
            if len(tracks)==1 and tracks[0]=='all':     tracks = self.__tracks__
            if len(features)==1 and features[0]=='all': features = self.__features__
            if verbose: print '|',
            for track in tracks:
                if 'moments' in features:
                    if not sm+'/'+rg+'/'+seq+'/'+track+'/moments/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                        print('hdf5 group: %s was not found'%sm+'/'+rg+'/'+seq.keys()[0]+'/'+track+'/moments/%s'%self.__window__)
                    else:
                        data = self.f[sm+'/'+rg+'/'+seq+'/'+track+'/moments/%s'%self.__window__]
                        self.read_attrs(data)
                        end = min(end,self.__len__) #correct any loads that are out of bounds
                        if self.__tile__:
                            #[1] set the chunk tiles including the last partial
                            tiles = [[i,i+self.__window__] for i in range(0,end-start,self.__window__)] #full windows
                            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
                            #[2] get the total container size, IE number of full (and partial) tiles
                            t = self.__len__/self.__window__
                            if  self.__len__%self.__window__>0: t += 1

                            a,b = (start/self.__window__),(start/self.__window__+len(tiles))
                            if self.__linear__:
                                self.O    = mp.Array(ctypes.c_double,self.__MN__*len(tiles),lock=False)
                                self.O[:] = data[a*self.__MN__:b*self.__MN__][:]
                            else:
                                self.O      = np.zeros((len(tiles),self.__MN__),dtype='f8')
                                self.O[:,:] = data[a:b,:]
                        else:
                            y = (end-start)-self.__window__
                            if self.__linear__:
                                self.O    = mp.Array(ctypes.c_double,self.__MN__*y,lock=False)
                                self.O[:] = data[start*self.__MN__:(end-self.__window__)*self.__MN__][:]
                            else:
                                self.O      = np.zeros((y,self.__MN__),dtype='f8')
                                self.O[:,:] = data[start:(end-self.__window__),:]
                        if verbose: print '%s:M:%s'%(track.replace('reads_',''),
                                                     [1 if i else 0 for i in [self.check()]][0]),
                if 'spectrum' in features:
                    if not sm+'/'+rg+'/'+seq+'/'+track+'/spectrum/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                        print('hdf5 group: %s was not found'%sm+'/'+rg+'/'+seq+'/'+track+'/spectrum/%s'%self.__window__)
                    else:
                        data = self.f[sm+'/'+rg+'/'+seq+'/'+track+'/spectrum/%s'%self.__window__]
                        self.read_attrs(data)
                        end = min(end,self.__len__) #correct any loads that are out of bounds
                        if self.__tile__:
                            #[1] set the chunk tiles including the last partial
                            tiles = [[i,i+self.__window__] for i in range(0,end-start,self.__window__)] #full windows
                            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
                            #[2] get the total container size, IE number of full (and partial) tiles
                            t = self.__len__/self.__window__
                            if  self.__len__%self.__window__>0: t += 1

                            a,b = (start/self.__window__),(start/self.__window__+len(tiles))
                            if self.__linear__:
                                self.O    = mp.Array(ctypes.c_float,self.__SN__*len(tiles),lock=False)
                                self.O[:] = data[a*self.__SN__:b*self.__SN__][:]
                            else:
                                self.O      = np.zeros((len(tiles),self.__SN__),dtype='f4')
                                self.O[:,:] = data[a:b,:]
                        else:
                            y = (end-start)-self.__window__
                            if self.__linear__:
                                self.O    = mp.Array(ctypes.c_float,self.__SN__*y,lock=False)
                                self.O[:] = data[start*self.__SN__:(end-self.__window__)*self.__SN__][:]
                            else:
                                self.O      = np.zeros((y,self.__SN__),dtype='f4')
                                self.O[:,:] = data[start:(end-self.__window__),:]
                        if verbose: print '%s:S:%s'%(track.replace('reads_',''),
                                                     [1 if i else 0 for i in [self.check()]][0]),
                if 'transitions' in features:
                    if not sm+'/'+rg+'/'+seq+'/'+track+'/transitions/%s'%self.__window__ in self.f:  #(2) check for the sm:rg:seq
                        print('hdf5 group: %s was not found'%sm+'/'+rg+'/'+seq+'/'+track+'/transitions/%s'%self.__window__)
                    else:
                        data = self.f[sm+'/'+rg+'/'+seq+'/'+track+'/transitions/%s'%self.__window__]
                        self.read_attrs(data)
                        end = min(end,self.__len__) #correct any loads that are out of bounds
                        if self.__tile__:
                            #[1] set the chunk tiles including the last partial
                            tiles = [[i,i+self.__window__] for i in range(0,end-start,self.__window__)] #full windows
                            if (end-start)%self.__window__>0: tiles[-1][1] = end-start
                            #[2] get the total container size, IE number of full (and partial) tiles
                            t = self.__len__/self.__window__
                            if  self.__len__%self.__window__>0: t += 1

                            a,b = (start/self.__window__),(start/self.__window__+len(tiles))
                            if self.__linear__:
                                self.O    = mp.Array(ctypes.c_float,(self.__TN__**2)*len(tiles),lock=False)
                                self.O[:] = data[a*(self.__TN__**2):b*(self.__TN__**2)][:]
                            else:
                                self.O        = np.zeros((len(tiles),self.__TN__,self.__TN__),dtype='f4')
                                self.O[:,:,:] = data[a:b,:,:]
                        else:
                            y = (end-start)-self.__window__
                            if self.__linear__:
                                self.O    = mp.Array(ctypes.c_float,(self.__TN__**2)*y,lock=False)
                                self.O[:] = data[start*(self.__TN__**2):(end-self.__window__)*(self.__TN__**2)][:]
                            else:
                                self.O        = np.zeros((y,self.__TN__,self.__TN__),dtype='f4')
                                self.O[:,:,:] = data[start:(end-self.__window__),:,:]
                        if verbose: print '%s:T:%s'%(track.replace('reads_',''),
                                                     [1 if i else 0 for i in [self.check()]][0]),
            self.f.close()
            if verbose: print('\nall reading completed from hdf5 container for %s-bp chunk'%(end-start))
            return True
        else:
            print('hdf5 container path not found: %s'%hdf5_path)
            return False

    #:::TO DO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def buffer_seq(self,hdf5_path,seq,tracks=['total'],features=['moments'],verbose=False):
        k = seq.keys()[0]
        passes = k/self.__chunk__
        last   = k%self.__chunk__
        if last == 0: last = []
        else:         last = [last]
        chunks = [self.__chunk__ for y in range(passes)]+last
        x = 0
        for i in range(len(chunks)):
            self.buffer_chunk(hdf5_path, seq, x, x+chunks[i],
                              tracks=['reads_all'],features=['moments'],verbose=False)
            x += chunks[i]
        return True
    #-------------------------------------------------------------------------------------------------------------------
    
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
