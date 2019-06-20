#!/usr/bin/env python
#Copyright (C) 2019 Timothy James Becker
#command line wrapper (and test script) for HFM class feature extraction of a bam file
#bam files can have multiple sample tags (SM) each of which can have multiple rg tags (RG)
#provisions for pooling reads at the read group level for each sample are provided

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
import subprocess32 as subprocess
#----------------------------
import multiprocessing as mp
from hfm import hfm
#----------------------------

des = """
HFM: Exact Hierarchical Feature Moment/Spectrum/Transition
Extraction for Analysis and Visualization
Batch Extractor Tool v0.1.0, Copyright (C) 2019 Timothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--in_path',type=str,help='sam/bam/cram file or input directory\t[None]')
parser.add_argument('-r', '--ref_path',type=str,help='for cram inputs\t[None]')
parser.add_argument('-o', '--out_dir',type=str,help='output directory\t[None]')
parser.add_argument('-p', '--cpus',type=int,help='number of parallel core||readers (pools to sequences)\t[1]')
parser.add_argument('-c', '--comp',type=str,help='compression type\t[lzf or gzip]')
parser.add_argument('-w', '--window',type=int,help='window size in bp\t[100]')
parser.add_argument('-b', '--branch',type=int,help='window branching factor\t[10]')
parser.add_argument('-t', '--tile',type=bool,help='use tiles for features as opposed to 1-bp sliding windows of size w\t[True]')
parser.add_argument('-s', '--seqs',type=str,help='comma seperated list of seqs that will be extracted, \t[all]')
v = 'comma seperated list of vectors that will be extracted for each seq, all gives every available\t[total]'
parser.add_argument('-v','--vectors',type=str,help=v)
f = 'comma seperated list of features that will be calculated for each vector on each sequence, all gives every available\t[moments]'
parser.add_argument('-f','--features',type=str,help=f)
args = parser.parse_args()

if args.in_path is not None:
    if args.in_path.endswith('.sam') or args.in_path.endswith('.bam') or args.in_path.endswith('.cram'):
        alignment_paths = [args.in_path]
    else:
        alignment_paths = glob.glob(args.in_path+'/*.sam')+glob.glob(args.in_path+'/*.bam')+glob.glob(args.in_path+'/*.cram')
    print('found files: %s'%alignment_paths)
else:
    print('no input directory was specified!\n')
    raise IOError

if args.out_dir is not None:
    hdf5_path = args.out_dir
else:
    print('no output path was specified!\n')
    raise IOError

if args.cpus is not None:     cpus = args.cpus
else:                         cpus = 1
if args.window is not None:   w    = args.window
else:                         w    = 100
if args.branch is not None:   w_b  = args.branch
else:                         w_b  = 10
if args.tile is not None:     tile = args.tile
else:                         tile = True
if args.seqs is not None:     seqs = args.seqs.split(',')
else:                         seqs = 'all'
if args.vectors is not None:  vect = args.vectors.split(',')
else:                         vect = ['total','discordant','orient_out','orient_um','orient_chr',
                                      'clipped','fwd_rev_diff','mapq','tlen','GC']
if args.features is not None: feat = args.features.split(',')
else:                         feat = ['moments']
if args.comp is not None: comp     = args.comp
else:                     comp     = 'gzip'


result_list = []
def collect_results(result):
    result_list.append(result)

#can call this in ||----------------------------------------------------------------
def process_seq(alignment_path,base_name,sms,seq,merge_rg=True,
                tracks=['total'],features=['moments'],window=100,window_branch=10,
                tile=True,tree=True,comp='gzip',verbose=False):
    result = ''
    start = time.time()
    try:
        s = hfm.HFM(tile=tile,window=window,window_branch=window_branch,window_root=int(1E9),compression=comp)
        s.extract_seq(alignment_path,base_name,sms,seq,merge_rg=merge_rg,
                      tracks=tracks,features=features,verbose=verbose)
        print('seq %s extracted, starting window updates'%seq[seq.keys()[0]])
        if tree: s.update_seq_tree(base_name,seq,verbose=verbose)
    except Exception as E:
        result = str(E)
        pass
    stop = time.time()
    return {seq[seq.keys()[0]]:stop-start,'result':result}

if not os.path.exists(hdf5_path): os.makedirs(hdf5_path)

for alignment_path in alignment_paths:
    if not os.path.exists(hdf5_path + '/seqs/'): os.makedirs(hdf5_path + '/seqs/')
    extension = '.'+alignment_path.rsplit('.')[-1]
    base_name = hdf5_path+'/seqs/'+alignment_path.rsplit('/')[-1].rsplit(extension)[0]
    hdf5_out  = base_name.replace('/seqs/','/')+'.merged.hdf5'
    seq_order  = hfm.get_seq(alignment_path) #a python list sorted by largest sequence to smallest: [{str(seq):int(len)}]

    S = []
    sms       = hfm.get_sm(alignment_path)  #this is all the read groups listed in the file
    if seqs != 'all':
        for i in range(len(seq_order)):
            if seq_order[i][seq_order[i].keys()[0]] in seqs:
                S += [{seq_order[i].keys()[0]:seq_order[i][seq_order[i].keys()[0]]}]
    else:
        for i in range(len(seq_order)):
            S += [{seq_order[i].keys()[0]: seq_order[i][seq_order[i].keys()[0]]}]
    print(S)

    #debug one seq here----------------------------------------------------------
    # s = hfm.HFM(tile=tile,window=w,window_branch=int(1E1),window_root=int(1E9), compression=comp)
    # s.extract_seq(alignment_path,base_name,sms,S[-1],merge_rg=True,tracks=vect,features=feat,verbose=True)
    # s.update_seq_tree(base_name,S[-1])
    #
    # self = s
    # sm = 'TCRBOA7_T'
    # rg = 'all'
    # seq = '21'
    # track = 'total'
    # hdf5_path = base_name
    # import ctypes
    # import numpy as np
    # import core
    # from h5py import File
    #debug one seq here----------------------------------------------------------

    t_start = time.time()
    p1 = mp.Pool(processes=cpus)
    for seq in S: #|| on seq
        p1.apply_async(process_seq,
                       args=(alignment_path,base_name,sms,seq,
                             True,vect,feat,w,w_b,tile,True,comp,True),
                       callback=collect_results)
        time.sleep(0.25)
    p1.close()
    p1.join()
    hfm.merge_seqs(hdf5_path+'/seqs/',hdf5_out) #merge the files
    print(subprocess.check_output(['rm','-rf',hdf5_path+'/seqs/'])) #delete the seperate files
    t_stop = time.time()
    print('sample %s || cython with %s cpus in %s sec'%(base_name,cpus,t_stop-t_start))
    #executiong of SAFE Test Pipeline for a sample:::::::::::::::::::::::::::::::::::::::
    if not all([l['result']=='' for l in result_list]):
        s = ''
        for l in result_list: s += l['result']
        with open(hdf5_path+'error','w') as f: f.write(s)