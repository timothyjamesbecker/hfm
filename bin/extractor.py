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
HFM: Exact Hierarchical Feature Moment/Spectrum/Transition
Extraction for Analysis and Visualization
Batch Extractor Tool """+str(hfm.core.__version__)+""", Copyright (C) 2019 Timothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--in_path',type=str,help='sam/bam/cram file or input directory\t[None]')
parser.add_argument('-r', '--ref_path',type=str,help='for cram inputs\t[None]')
parser.add_argument('-o', '--out_dir',type=str,help='output directory\t[None]')
parser.add_argument('-p', '--cpus',type=int,help='number of parallel core||readers (pools to sequences)\t[1]')
parser.add_argument('-c', '--comp',type=str,help='compression type\t[lzf or gzip]')
parser.add_argument('-n', '--no_merge_rg',action='store_true',help='do not merge all rg into one called "all"\t[False]')
parser.add_argument('-w', '--window',type=int,help='window size in bp\t[100]')
parser.add_argument('-b', '--branch',type=int,help='window branching factor\t[10]')
parser.add_argument('--bins',type=str,help='comma-seperated bin counting boundries for spectrum and transition features\t[None]')
parser.add_argument('--slide',action='store_true',help='use 1-bp sliding windows of size w for features as opposed to tiles\t[False]')
parser.add_argument('-s', '--seqs',type=str,help='comma seperated list of seqs that will be extracted, \t[all]')
v = 'comma seperated list of vectors that will be extracted for each seq, all gives every available\t[total]'
parser.add_argument('-v','--vectors',type=str,help=v)
f = 'comma seperated list of features that will be calculated for each vector on each sequence, all gives every available\t[moments]'
parser.add_argument('-f','--features',type=str,help=f)
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
    print('using advanced reprocessing mode for rebranching and retransforming base windows of existing hfm data...')
    alignment_paths = []
    hdf5_reproc_path = args.reproc_dir
    hdf5_path = sorted(list(set(glob.glob(args.out_dir+'/*.hdf5')).difference(set(glob.glob(args.out_dir+'/*.reproc.hdf5')))))
    print('located the following hfm files for reprocessing: %s'%hdf5_path)
    if len(hdf5_path)<1:
        print('using advanced reprocessing mode without any hfm data files...')
        raise IOError
else:
    print('no input directory was specified!\n')
    raise IOError
if args.cpus is not None:     cpus = args.cpus
else:                         cpus = 1
if args.no_merge_rg:          merge_rg = False
else:                         merge_rg = True
if args.window is not None:   w    = args.window
else:                         w    = 100
if args.branch is not None:   w_b  = args.branch
else:                         w_b  = 10
if args.slide is not None:    tile = not args.slide
else:                         tile = True
if args.bins is not None:     bins = sorted(list(set([int(x) for x in args.bins.rsplit(',')])))
else:                         bins = list(np.arange(0.0,1.0+1/25,1/25))
if args.no_clean is None:     end_clean = args.no_clean
else:                         end_clean = True
if args.seqs is not None:     seqs = args.seqs.split(',')
else:                         seqs = 'all'
if args.vectors is not None:  vect = args.vectors.split(',')
else:                         vect = ['total','primary','discordant','orient_out','orient_same','orient_um','orient_chr','deletion','insertion',
                                      'right_clipped','left_clipped','fwd_rev_diff','mapq_pp','mapq_dis','tlen_pp','tlen_dis','GC']
if args.features is not None: feat = args.features.split(',')
else:                         feat = ['moments']
if args.comp is not None: comp     = args.comp
else:                     comp     = 'gzip'

# || return data structure: async queue
result_list = []
def collect_results(result):
    result_list.append(result)

#can call this in ||----------------------------------------------------------------
def process_seq(alignment_path,base_name,sms,seq,merge_rg=True,
                tracks=['total'],features=['moments'],window=100,window_branch=10,
                tile=True,tree=True,bins=None,comp='gzip',verbose=False):
    result = ''
    start = time.time()
    samples = list(set([sms[k] for k in sms]))
    chunk = max(window,window*(int(1E6/len(sms))/window))
    try:
        s = hfm.HFM(tile=tile,window=window,window_branch=window_branch,
                    window_root=int(1E9),bins=bins,chunk=chunk,compression=comp)
        s.extract_seq(alignment_path,base_name,sms,seq,merge_rg=merge_rg,
                      tracks=tracks,features=features,verbose=verbose)
        print('seq %s extracted, starting window updates'%seq[list(seq.keys())[0]])
        if tree: s.update_seq_tree(base_name,seq,verbose=verbose)
    except Exception as E:
        result = str(E)
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
            print('starting %s samples with %s total rgs and merge_rg=%s\nv=%s\nf=%s\nw=%s\nb=%s\ntile=%s'%\
                  (len(set(sms.values())),len(sms),merge_rg,vect,feat,w,w_b,tile))
            for seq in S: #|| on seq
                if not os.path.exists(base_name+'.seq.'+seq[list(seq.keys())[0]]+'.hdf5'):
                    p1.apply_async(process_seq,
                                   args=(alignment_path,base_name,sms,seq,
                                         merge_rg,vect,feat,w,w_b,tile,True,bins,comp,True),
                                   callback=collect_results)
                    time.sleep(0.25)
            p1.close()
            p1.join()
            print('finished processing individual seqs')
            if all([l['result']=='' for l in result_list]) and len(glob.glob(hdf5_path + '/seqs/*.hdf5')) >= len(S):
                print('merging and cleaning intermediary files')
                if len(glob.glob(args.out_dir+'/seqs/*temp*.hdf5'))>0:
                    print(subprocess.check_output(' '.join(['rm',args.out_dir+'/seqs/*temp*.hdf5']),shell=True))
                # hfm.merge_seqs(hdf5_path+'/seqs/',hdf5_out) #merge the files
                #print(subprocess.check_output(' '.join(['rm','-rf',hdf5_path+'/seqs/']),shell=True)) #delete the seperate files
            else:
                s = ''
                for l in result_list: s += l['result']+'\n'
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
for s in seq_order[0:1]:
    if s[s.keys()[0]] in args.seqs.rsplit(','):
        seq = {s.keys()[0]:s[s.keys()[0]]}
tracks,features,verbose = vect,feat,True
extension = '.'+alignment_path.rsplit('.')[-1]
base_name = hdf5_path+'/seqs/'+alignment_path.rsplit('/')[-1].rsplit(extension)[0]
hdf5_out_path = base_name+'.seq.'+seq[seq.keys()[0]]+'.hdf5'
sms = hfm.get_sam_sm(alignment_path)
samples = list(set([sms[k] for k in sms]))
chunk = max(w,w*(int(1E6/len(sms))/w))
s = hfm.HFM(tile=tile,window=w,window_branch=w_b,window_root=int(1E9),
            chunk=chunk,compression='gzip')
s.extract_seq(alignment_paths[0],base_name,sms,
              seq,merge_rg=merge_rg,tracks=tracks,features=features,verbose=verbose)
# print('seq %s extracted, starting window updates'%seq[seq.keys()[0]])
# s.update_seq_tree(base_name,seq,verbose=verbose)
hdf5_path = base_name
import numpy as np
from h5py import File
import core
import ctypes
self = s
#debug one seq here----------------------------------------------------------
"""