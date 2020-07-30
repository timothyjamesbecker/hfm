#!/usr/bin/env python

#TESTING INFO:
#./data directory tested with: 
#(1) low and high coverage WGS and WES at 1000 Genomes bam files: ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/
#(2) multiple-readgroup DNA-seq (Illumina) bam file used in the GenomeSTRiP data test folder:
#http://software.broadinstitute.org/software/genomestrip/download-genome-strip
#(3) ENCODE data at UCSC: CHIAPET, CHIPSEQ, RNASEQ dataset

import argparse
import os
import time
import glob
import sys
import numpy as np
if sys.version_info<(3,0):
    import subprocess32 as subprocess
else:
    import subprocess
#----------------------------
import multiprocessing as mp
from hfm import hfm
#----------------------------

des = """
HFE: Hierarchical [moment/spectrum/transition] Feature Extraction
Multi-SAM|BAM|CRAM Batch Extractor Tool """+str(hfm.core.__version__)+""", Copyright (C) 2019-2020 Timothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--in_path',type=str,help='sam/bam/cram file or input directory\t[None]')
parser.add_argument('--ref_path',type=str,help='for cram inputs and realignment based features\t[None]')
parser.add_argument('--out_dir',type=str,help='output directory\t[None]')
filter_help = """semi colon seperated then comma seperated per track pre-filter parameter specification\t[None]
[syntax] trk:name,value,width,over,mix
[EX-1] MD:db4,0.5,1E4,0,0.2;
[EX-2] tlen_dis_rd:poly,1.0,1E9,0,1.0;left_smap_same:poly,1.0,1E9,0,1.0;right_smap_same:poly,1.0,1E9,0,1.0;
"""
parser.add_argument('--filter',type=str,help=filter_help)
parser.add_argument('--window',type=int,help='window size in bp\t[100]')
parser.add_argument('--branch',type=int,help='window branching factor\t[10]')
parser.add_argument('--chunk',type=float,help='chunk size in bp per cpu (controls MEM-use/speed)\t[10E6]')
parser.add_argument('--no_mem_map',action='store_true',help='buffer with memory mapped arrays\t[True]')
parser.add_argument('--min_smapq',type=int,help='minimum split mapq value for realigned clipped read fragment\t[20]')
parser.add_argument('--bins',type=str,help='comma-seperated bin counting boundries for spectrum and transition features\t[None]')
parser.add_argument('--no_merge_rg',action='store_true',help='do not merge all rg into one called "all"\t[False]')
parser.add_argument('--slide',action='store_true',help='use 1-bp sliding windows of size w for features as opposed to tiles\t[False]')
parser.add_argument('--dna',action='store_true',help='create nucleotide transition tracks\t[False]')
parser.add_argument('--sub',action='store_true',help='exact ref_seq matching for substitution track\t[False]')
parser.add_argument('--seqs',type=str,help='comma seperated list of seqs that will be extracted, \t[all in BAM header]')
trks  = ['total','primary','alternate','proper_pair','discordant','RD','GC','MD',
         'mapq_pp','mapq_dis','big_del','deletion','insertion','substitution','splice','fwd_rev_diff',
         'tlen_pp', 'tlen_pp_rd', 'tlen_dis', 'tlen_dis_rd','right_clipped','left_clipped',
         'orient_same','orient_out','orient_um','orient_chr',
         'left_smap_same','left_smap_diff','right_smap_same','right_smap_diff']
t_help = 'comma seperated list of tracks that will be extracted for each seq, all gives every available\t[%s]'%','.join(trks)
parser.add_argument('--tracks',type=str,help=t_help)
f_help = 'comma seperated list of features that will be calculated for each track on each sequence, all gives every available\t[moments]'
parser.add_argument('--features',type=str,help=f_help)
parser.add_argument('--cpus',type=int,help='number of parallel core||readers (pools to sequences)\t[1]')
parser.add_argument('--comp',type=str,help='hdf5 block compression type\t[faster and larger lzf or slower and smaller gzip-9]')
parser.add_argument('--test',action='store_true',help='will run the multisample.bam test file and save result in the out_dir')
parser.add_argument('--reproc_dir',type=str,help='output directory for rebranching and retransforming hfm data from base windows\t[None]')
parser.add_argument('--no_clean',action='store_true',help='will not check and cleans end points for rebranching operations\t[False]')
args = parser.parse_args()

if args.out_dir is not None:
    hdf5_path = args.out_dir
else:
    print('no output path was specified!\n')
    raise IOError
if args.in_path is not None:
    if args.in_path.endswith('.sam') or args.in_path.endswith('.bam') or args.in_path.endswith('.cram'):
        alignment_paths = [args.in_path]
    else:
        alignment_paths = glob.glob(args.in_path+'/*.sam')+glob.glob(args.in_path+'/*.bam')+glob.glob(args.in_path+'/*.cram')
    print('found files: %s'%alignment_paths)
    hdf5_reproc_path = None
elif args.test:
    print('no input was specified, but the test options were set for multisample.bam to be processed')
    alignment_paths = [os.path.dirname(os.path.abspath(hfm.__file__)) + '/data/multisample.bam']
    hdf5_reproc_path = None
elif (args.out_dir is not None) and (args.reproc_dir is not None) and (args.in_path is None): #have and out_dir and reproc_dir
    print('using reprocessing mode for rebranching and retransforming base windows of existing base hfm data...')
    alignment_paths = []
    hdf5_reproc_path = args.reproc_dir
    hdf5_path = sorted(list(set(glob.glob(args.out_dir+'/*.hdf5')).difference(set(glob.glob(args.out_dir+'/*.reproc.hdf5')))))
    print('located the following hfm files for reprocessing: %s'%hdf5_path)
    if len(hdf5_path)<1:
        print('trying to use reprocessing mode without base hfm data files...')
        raise IOError
else:
    print('no input directory was specified!\n')
    raise IOError
if args.ref_path is None:     print('no reference fasta path was specified: realignment and substitution features are disabled...')
if args.cpus is not None:     cpus = args.cpus
else:                         cpus = 1
if args.no_merge_rg:          merge_rg = False
else:                         merge_rg = True
if args.no_mem_map:           mem_map  = None
else:                         mem_map  = hdf5_path+'/seqs/'
if args.window is not None:   w     = args.window
else:                         w     = 100
if args.branch is not None:   w_b   = args.branch
else:                         w_b   = 10
if args.chunk is not None:    chunk = int(args.chunk)
else:                         chunk = int(1E9)
if args.min_smapq is not None:min_smapq = args.min_smapq
else:                         min_smapq = 20
if args.slide is not None:    tile  = not args.slide
else:                         tile  = True
if args.bins is not None:     bins  = sorted(list(set([int(x) for x in args.bins.rsplit(',')])))
else:                         bins  = list(np.arange(0.0,1.0+1/25,1/25))
if args.no_clean is None:     end_clean = args.no_clean
else:                         end_clean = True
if args.seqs is not None:     seqs  = args.seqs.split(',')
else:                         seqs  = 'all'
if args.tracks is not None:
    U = []
    u_trks  = args.tracks.split(',')
    for trk in u_trks:
        if trk in trks: U += [trk]
    trks = U
if args.dna:                  trks += ['A-A','A-C','A-G','A-T','C-A','C-C','C-G','C-T',
                                       'G-A','G-C','G-G','G-T','T-A','T-C','T-G','T-T']
if args.features is not None: feats = args.features.split(',')
else:                         feats = ['moments']
if args.filter is not None: # trk:flt_name,value,width,over,mix; ... [EX] total:db4,0.5,int(1E9),0.0,1.0
    fltr_params = {}
    raw_params = args.filter.split(';')
    for param in raw_params:
        trk = param.split(':')[0]
        prms = param.split(':')[-1].split(',')
        if trk in trks and len(prms)==5:
            fltr_params[trk] = {'type':prms[0],'value':float(prms[1]),'width':int(float(prms[2])),
                                'over':int(float(prms[3])),'mix':float(prms[4])}
    if len(fltr_params)<1:
        print('filter parameter syntax error:%s'%args.filter)
        fltr_params = None #error parsing inputs
    else: print(fltr_params)
else:
    flt_trks = {'tlen_dis_rd':    {'type':'poly','value':0.5,'width':int(1E9),'over':0,'mix':1.0},
                'left_smap_same': {'type':'poly','value':0.5,'width':int(1E9),'over':0,'mix':1.0},
                'left_smap_diff': {'type':'poly','value':0.5,'width':int(1E9),'over':0,'mix':1.0},
                'right_smap_same':{'type':'poly','value':0.5,'width':int(1E9),'over':0,'mix':1.0},
                'right_smap_diff':{'type':'poly','value':0.5,'width':int(1E9),'over':0,'mix':1.0}}
    fltr_params = {}
    for flt in flt_trks:
        if flt in trks:
            fltr_params[flt] = flt_trks[flt]
    if len(fltr_params)<1: fltr_params = None
if args.comp is not None: comp      = args.comp
else:                     comp      = 'gzip'

# || return data structure: async queue
result_list = []
def collect_results(result):
    result_list.append(result)

#can call this in ||----------------------------------------------------------------
def process_seq(alignment_path,base_name,sms,seq,merge_rg=True,exact_sub=False,
                tracks=['total'],features=['moments'],filter_params=None,window=100,window_branch=10,
                chunk=int(10E6),window_root=int(1E9),tile=True,tree=True,bins=None,comp='gzip',
                ref_path=None,min_smapq=20,mem_map=None,mm_chunk=int(3E8),no_mm_chunk=int(10E6),verbose=False):
    result = ''
    start = time.time()
    samples = list(set([sms[k] for k in sms]))
    corrected_chunk = int(max(window,window*(int(chunk//len(sms))//window)))
    mem_map_tracks = ['tlen_dis_rd','tlen_pp_rd','left_smap_same','left_smap_diff','right_smap_same','right_smap_diff']
    mem_map_tracks = sorted(list(set(mem_map_tracks).intersection(set(tracks))))
    try:
        # need to do discordant and propper pair for tlen_pp_rd and tlen_dis_rd
        if len(mem_map_tracks)>0:
            corrected_mem_chunk    = int(max(window,mm_chunk))
            corrected_no_mem_chunk = int(max(window,window*(int(no_mm_chunk//len(sms))//window)))
            add_on_tracks = ['discordant','proper_pair','tlen_dis','tlen_pp','mapq_dis','mapq_pp']

            #first do the tracks that don't need memory mapping accross the sequence------------------------
            no_mem_map_tracks = sorted(list(set(tracks).difference(mem_map_tracks+add_on_tracks)))
            h = hfm.HFM(tile=tile,window=window,window_branch=window_branch,
                    window_root=window_root,bins=bins,chunk=corrected_mem_chunk,compression=comp)
            print('built the hfm object for seq=%s'%seq)
            h.extract_seq(alignment_path,base_name,sms,seq,merge_rg=merge_rg,exact_sub=exact_sub,
                              tracks=no_mem_map_tracks,features=features,filter_params=filter_params,
                              ref_path=ref_path,min_smapq=min_smapq,mem_map_path=None,verbose=verbose)

            #now complete the tracks and their dependancies that need memory mapping accross the sequence---
            mem_map_tracks = sorted(list(set(mem_map_tracks+add_on_tracks).difference(tracks)))
            h = hfm.HFM(tile=tile,window=window,window_branch=window_branch,
                    window_root=window_root,bins=bins,chunk=corrected_no_mem_chunk,compression=comp)
            print('built the hfm object for seq=%s'%seq)
            h.extract_seq(alignment_path,base_name,sms,seq,merge_rg=merge_rg,exact_sub=exact_sub,
                              tracks=mem_map_tracks,features=features,filter_params=filter_params,
                              ref_path=ref_path,min_smapq=min_smapq,mem_map_path=mem_map,verbose=verbose)

        else:
            h = hfm.HFM(tile=tile,window=window,window_branch=window_branch,
                    window_root=window_root,bins=bins,chunk=corrected_chunk,compression=comp)
            print('built the hfm object for seq=%s'%seq)
            h.extract_seq(alignment_path,base_name,sms,seq,merge_rg=merge_rg,exact_sub=exact_sub,
                          tracks=tracks,features=features,filter_params=filter_params,
                          ref_path=ref_path,min_smapq=min_smapq,mem_map_path=None,verbose=verbose)
        print('seq %s extracted, starting window updates'%seq[list(seq.keys())[0]])
        if tree and window_branch>1: h.update_seq_tree(base_name,seq,verbose=verbose)
    except Exception as E:
        result = E
        pass
    stop = time.time()
    return {seq[list(seq.keys())[0]]:stop-start,'result':result}

#can call this in ||---------------------------------------------------------------------------------
def reprocess_seq(hdf5_in,hdf5_out,seq,window_branch,tree=True,end_clean=True,comp='gzip',verbose=False):
    result = ''
    start = time.time()
    try:
        print('reprocessing seq %s'%seq[list(seq.keys())[0]])
        if tree:
            if end_clean:
                sample = hdf5_in.rsplit('/')[-1].rsplit('.')[0]
                hdf5_temp = '/'.join(hdf5_in.rsplit('/')[:-1])+'/seqs/%s.temp.%s.hdf5'%(sample,seq[list(seq.keys())[0]])
                print('generating intermediate file %s'%hdf5_temp)
                hfm.HFM().clean_end_seq(hdf5_in,hdf5_temp,seq,comp,10,verbose)
                hfm.HFM().rebranch_update_seq_tree(hdf5_temp,hdf5_out,seq,window_branch,comp,verbose)
            else:
                print('recalculating seq %s updates'%seq[list(seq.keys())[0]])
                hfm.HFM().rebranch_update_seq_tree(hdf5_in,hdf5_out,seq,window_branch,comp,verbose)
    except Exception as E:
        result = str(E)
        pass
    stop  = time.time()
    return {seq[seq.keys()[0]]:stop-start,'result':result}

if type(hdf5_path)==str:
    if not os.path.exists(hdf5_path): os.makedirs(hdf5_path)
    for alignment_path in alignment_paths:
        if not os.path.exists(hdf5_path + '/seqs/'): os.makedirs(hdf5_path + '/seqs/')
        extension = '.'+alignment_path.rsplit('.')[-1]
        base_name = hdf5_path+'/seqs/'+alignment_path.rsplit('/')[-1].rsplit(extension)[0]
        hdf5_out  = base_name.replace('/seqs/','/')+'.merged.hdf5'
        seq_order  = hfm.get_sam_seq(alignment_path) #a python list sorted by largest sequence to smallest: [{str(seq):int(len)}]
        if os.path.exists(hdf5_out):
            print('alignment : %s has already been extracted as %s'%(alignment_path,hdf5_out))
        else:
            S = []
            sms       = hfm.get_sam_sm(alignment_path)  #this is all the read groups listed in the file
            if seqs != 'all':
                for i in range(len(seq_order)):
                    if seq_order[i][list(seq_order[i].keys())[0]] in seqs:
                        S += [{list(seq_order[i].keys())[0]:seq_order[i][list(seq_order[i].keys())[0]]}]
            else:
                for i in range(len(seq_order)):
                    S += [{list(seq_order[i].keys())[0]: seq_order[i][list(seq_order[i].keys())[0]]}]
            print(S)

            t_start = time.time()
            p1 = mp.Pool(processes=cpus)
            print('determined the following rgs: %s'%sms)
            print('starting %s samples with %s total rgs and merge_rg=%s\nt=%s\nf=%s\nw=%s\nb=%s\ntile=%s'%\
                  (len(set(sms.values())),len(sms),merge_rg,trks,feats,w,w_b,tile))
            for seq in S: #|| on seq
                if not os.path.exists(base_name+'.seq.'+seq[list(seq.keys())[0]]+'.hdf5'):
                    seq_args = (alignment_path,base_name,sms,seq,
                                merge_rg,args.sub,trks,feats,fltr_params,
                                w,w_b,chunk,int(1E9),tile,True,bins,
                                comp,args.ref_path,min_smapq,mem_map,int(3E8),int(10E6),True)
                    p1.apply_async(process_seq,args=seq_args,callback=collect_results)
                    time.sleep(0.25)
            p1.close()
            p1.join()
            print('finished processing individual seqs')
            if all([l['result']=='' for l in result_list]) and len(glob.glob(hdf5_path + '/seqs/*.hdf5')) >= len(S):
                print('merging and cleaning intermediary files')
                if len(glob.glob(args.out_dir+'/seqs/*temp*.hdf5'))>0:
                    print(subprocess.check_output(' '.join(['rm',args.out_dir+'/seqs/*temp*.hdf5']),shell=True))
                hfm.merge_seqs(hdf5_path+'/seqs/',hdf5_out) #merge the files
                print(subprocess.check_output(' '.join(['rm','-rf',hdf5_path+'/seqs/']),shell=True)) #delete the seperate files
            else:
                s = ''
                for l in result_list: s += str(l['result'])+'\n'
                with open(base_name+'.error','w') as f: f.write(s)
            t_stop = time.time()
            print('sample %s || cython with %s cpus in %s sec'%(base_name,cpus,t_stop-t_start))

elif type(hdf5_path)==list:
    for hdf5_in in hdf5_path:
        if not os.path.exists(args.out_dir+'/seqs/'): os.makedirs(args.out_dir+'/seqs/')
        base_name = hdf5_in.rsplit('/')[-1].rsplit('.')[0]
        hdf5_final_out = args.out_dir+'/'+base_name+'.reproc.hdf5'
        meta = hfm.HFM().get_base_window_attributes(hdf5_in)
        if 'window' in meta: w = meta['window']
        seqs = hfm.HFM().get_seqs(hdf5_in) #these are {seq:len}
        S = sorted([{seqs[k]:k} for k in seqs], key=lambda x: list(x.keys())[0],reverse=True)
        if os.path.exists(hdf5_final_out):
            print('hfm file %s already reprocessed as %s'%(hdf5_in,hdf5_final_out))
        else:
            t_start = time.time()
            p1 = mp.Pool(processes=cpus)
            for seq in S: #|| on seq
                hdf5_out = args.out_dir+'/seqs/%s.%s.hdf5'%(base_name,seq[list(seq.keys())[0]])
                p1.apply_async(reprocess_seq,
                               args=(hdf5_in,hdf5_out,seq,w_b,True,end_clean,comp,True),
                               callback=collect_results)
                time.sleep(0.25)
            p1.close()
            p1.join()
            if all([l['result']=='' for l in result_list]) and len(glob.glob(args.out_dir+'/seqs/*.hdf5')) >= len(S):
                if len(glob.glob(args.out_dir + '/seqs/*temp*.hdf5')) > 0:
                    print(subprocess.check_output(' '.join(['rm',args.out_dir+'/seqs/*temp*.hdf5']),shell=True))
                hfm.merge_seqs(args.out_dir+'/seqs/',hdf5_final_out) #merge the files
                print(subprocess.check_output(['rm','-rf',args.out_dir+'/seqs/'])) #delete the seperate files
            else:
                s = ''
                for l in result_list: s += l['result']
                with open(args.out_dir+'/%s.error'%base_name,'w') as f: f.write(s)
            t_stop  = time.time()
            print('sample %s || cython with %s cpus in %s sec' % (base_name,cpus,t_stop-t_start))

"""
#debug one seq here----------------------------------------------------------
if not os.path.exists(hdf5_path + '/seqs/'): os.makedirs(hdf5_path + '/seqs/')
seq = {}
alignment_path = alignment_paths[0]
seq_order = hfm.get_sam_seq(alignment_path)
for s in seq_order:
    if s[list(s.keys())[0]] in args.seqs.rsplit(','):
        seq = {list(s.keys())[0]:s[list(s.keys())[0]]}
tracks,features,verbose = vect,feat,True
extension = '.'+alignment_path.rsplit('.')[-1]
base_name = hdf5_path+'/seqs/'+alignment_path.rsplit('/')[-1].rsplit(extension)[0]
hdf5_out_path = base_name+'.seq.'+seq[list(seq.keys())[0]]+'.hdf5'
sms = hfm.get_sam_sm(alignment_path)
samples = list(set([sms[k] for k in sms]))
chunk = max(w,w*(int(1E6/len(sms))//w))
s = hfm.HFM(tile=tile,window=w,window_branch=w_b,window_root=int(1E9),
            chunk=chunk,compression='gzip')
hdf5_path = base_name
import numpy as np
from h5py import File
import core
import ctypes
self = s
s.extract_seq(alignment_path,base_name,sms,
              seq,merge_rg=merge_rg,tracks=tracks,features=features,verbose=verbose)
# print('seq %s extracted, starting window updates'%seq[seq.keys()[0]])
s.update_seq_tree(base_name,seq,verbose=verbose)
#debug one seq here----------------------------------------------------------
"""