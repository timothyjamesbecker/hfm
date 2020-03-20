#!/usr/bin/env python
import argparse
import glob
import os
import sys
import time
if sys.version_info<(3,0): import subprocess32 as subprocess
else: import subprocess
import multiprocessing as mp

def compress_fastq(fastq_path,fastq_final):
    l_start = time.time()
    command = ['gzip -9 -c %s > %s'%(fastq_path,fastq_final)]
    try: out = subprocess.check_output(' '.join(command),shell=True)
    except Exception as E: out = str(E)
    if len(glob.glob(fastq_path))>0:
        command = ['rm',fastq_path]
        try: out += subprocess.check_output(' '.join(command),shell=True)
        except Exception as E: out += str(E)
    l_stop = time.time()
    return [out,round(l_stop-l_start,2)]

def convert_am_to_fastq(sbcram,out_dir,mem,bio_tools):
    l_start = time.time()
    command = ['java -Xmx%sg -Dsamjdk.use_async_io_read_samtools=false -Dsamjdk.use_async_io_write_samtools=true'%mem,
               '-Dsamjdk.use_async_io_write_tribble=false -Dsamjdk.compression_level=2',
               '-jar %s/gatk-package-4.1.5.0-local.jar SamToFastq'%bio_tools,'--INPUT=%s'%sbcram,'--OUTPUT_PER_RG=true',
               '--MAX_RECORDS_IN_RAM=1000000','--TMP_DIR=%s'%out_dir,'--OUTPUT_DIR=%s'%out_dir]
    try: out = subprocess.check_output(' '.join(command),shell=True)
    except Exception as E: out = str(E)
    l_stop = time.time()
    return [out,round(l_stop-l_start,2)]

# || return data structure: async queue
result_list = []
def collect_results(result):
    result_list.append(result)

des = """
GATK4/PicardTools FASTQ read extraction from a SAM/BAM/CRAM directory followed by gzip -9 compression
, Copyright (C) 2020 Timothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--in_path',type=str,help='sam/bam/cram file or input directory\t[None]')
parser.add_argument('-o', '--out_dir',type=str,help='fastq output directory\t[None]')
parser.add_argument('-f', '--final_dir',type=str,help='final fastq.gz directory\t[None]')
parser.add_argument('-M', '--mem',type=int,help='memory to allocate\t[None]')
parser.add_argument('-S', '--samples',type=int,help='number of samples to read from\t[None]')
parser.add_argument('-G', '--gzip',type=int,help='number of gzip decompressions to use\t[None]')
parser.add_argument('-t', '--bio_tools',type=str,help='path to the gatk/picardtools binary\t[]')
args = parser.parse_args()

if args.in_path is not None:
    in_path = args.in_path
    ext = in_path.endswith('.sam') or in_path.endswith('.bam') or in_path.endswith('.cram')
    if ext: sbcrams = [in_path]
    else:   sbcrams = glob.glob(in_path+'/*.sam')+glob.glob(in_path+'/*.bam')+glob.glob(in_path+'/*.cram')
    sbcrams = sorted(sbcrams)
else: raise IOError
if args.out_dir is not None:
    out_dir = args.out_dir
else:
    if ext: out_dir = '/'.join(in_path.rsplit('/')[:-1])+'/'
    else:   out_dir = in_path+'/'
if not os.path.exists(out_dir): os.mkdir(out_dir)
if args.final_dir is not None:
    final_dir = args.final_dir
else:
    final_dir = out_dir
if not os.path.exists(final_dir): os.mkdir(final_dir)
if args.bio_tools is not None:
    bio_tools = args.bio_tools
else:
    bio_tools = ''
if args.mem is not None:
    mem = args.mem
else:
    mem = 16
if args.samples is not None:
    samples = args.samples
else:
    samples = 1
if args.gzip is not None:
    gzip = args.gzip
else:
    gzip = 4

if __name__=='__main__':
    t_start = time.time()
    jobs = []
    for i in range(0,len(sbcrams),samples):
        jobs += [[sbcrams[j] for j in range(i,min(i+samples,len(sbcrams)),1)]]
    p1 = mp.Pool(processes=samples)
    p2 = mp.Pool(processes=gzip)
    for job in jobs:
        print('starting job:%s'%job)
        for sbcram in job:
            p1.apply_async(convert_am_to_fastq,
                           args=(sbcram,out_dir,mem,bio_tools),
                           callback=collect_results)
            time.sleep(0.25)
        p1.close()
        p1.join()
        print('finished %s conversions'%len(job))
        gzips = glob.glob(out_dir+'/*.fastq')
        for fastq_path in gzips:
            fastq_final = final_dir+'/'+fastq_path.rsplit('/')[1]
            p2.apply_async(compress_fastq,
                           args=(fastq_path,fastq_final),
                           callback=collect_results)
            time.sleep(0.25)
        p2.close()
        p2.join()
        print('finished %s gzips'%len(gzips))
    results = []
    for result in result_list: results += [result]
    print(results)
    t_stop = time.time()
    print('file=%s completed in %s sec'%(sbcram,round(t_stop-t_start,2)))
