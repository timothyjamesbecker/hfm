#!/usr/bin/env python
import argparse
import time
import sys
import os
import glob
import numpy as np
import multiprocessing as mp
import subprocess32 as subprocess

des = """1000 genomes ftp download management script"""
parser = argparse.ArgumentParser(description=des)
parser.add_argument('-l', '--ftp_list', type=str, help='path to the tsv ftp download list file\t[None]')
parser.add_argument('-p', '--pop_list', type=str, help='path to the tsv population read group file\t[None]')
parser.add_argument('-o', '--out_dir', type=str, help='output directory to save ...bam/ into\t[None]')
parser.add_argument('-r', '--reference', type=str, help='reference fasta file\t[None]')
parser.add_argument('-c', '--connections', type=int, help='number of connections to use at once\t[1]')
parser.add_argument('-t', '--threads', type=int, help='number of threads per connection\t[1]')
parser.add_argument('-m', '--memory', type=int, help='total GB to allocate per connection   \t[1]')
parser.add_argument('-s', '--num_samples', type=str, help='total number of samples to download\t[1]')
parser.add_argument('--high_cov', action='store_true', help='get the high coverage data\t[False]')
args = parser.parse_args()

if args.ftp_list is not None:
    sample_ftp_path = args.ftp_list
else:
    raise IOError
if args.pop_list is not None:
    sample_list_path = args.pop_list
else:
    raise IOError
if args.out_dir is not None:
    out_dir = args.out_dir
    if not os.path.exists(out_dir): os.makedirs(out_dir)
else:
    raise IOError
if args.reference is not None and os.path.exists(args.reference):
    ref = args.reference
else:
    raise IOError
if args.connections is not None:
    cpus = args.connections
else:
    cpus = 1
if args.threads is not None:
    threads = args.threads
else:
    threads = 3
if args.memory is not None:
    memory = args.memory
else:
    memory = 2
if args.num_samples is not None and args.num_samples.upper() != 'ALL':
    num_samples = int(args.num_samples)
elif args.num_samples.upper() == 'ALL':
    sample_list, sample_ftp = [], []
    with open(sample_list_path, 'r') as f:
        sample_list = f.readlines()
    sample_list = [s.replace('\n', '').split('\t') for s in sample_list]
    with open(sample_ftp_path, 'r') as f:
        sample_ftp = f.readlines()
    sample_ftp = [s.replace('\n', '') for s in sample_ftp]
    P,N,RG = {},{},{}
    for sample in sample_list:
        if sample[0] in sample_ftp:
            if P.has_key(sample[1]):
                P[sample[1]] += [sample[0]]
            else:
                P[sample[1]] = [sample[0]]
            RG[sample[0]] = ['@RG\tID:%s\tSM:%s\tPL:%s\tLB:%s\tPU:%s'%\
                             (rg.split(';')[0],sample[0],rg.split(';')[2],rg.split(';')[1],rg.split(';')[0]) for rg in sample[2].split(',')]
        else:
            if N.has_key(sample[1]):
                N[sample[1]] += [sample[0]]
            else:
                N[sample[1]] = [sample[0]]
    num_samples = sum([len(P[k]) for k in P])
else:
    num_samples = 1

def associate_bas_rgs(bas_dir):
    B,bas = {},glob.glob(bas_dir+'/*.bas')
    for b in bas:
        with open(b,'r') as f: data = [i.replace('\n','').split('\t') for i in f.readlines()]
        if len(data)>0: #header should be: 0:'bam_filename', 1:'md5', 2:'study', 3:'sample', 4:'platform', 5:'library', 6:'readgroup',...
            for i in range(1,len(data),1):
                sample,platform,library,readgroup = data[i][3],data[i][4],data[i][5],data[i][6]
                if sample in B: B[sample] += [[readgroup,library,platform]]
                else:           B[sample]  = [[readgroup,library,platform]]
    return B

# wget from the ftp a specified sample, unless it is already in the download.check file
def wget_bam_realign(base_url,log_path,ref_mmi,sample):
    print('starting sample %s' % sample)
    output, err = '', ''
    # [1] download unmapped index
    url = base_url + '/%s/alignment/%s.unmapped*.bam.bai' % (sample, sample)
    print(url)
    command = ['cd', '/'.join(log_path.rsplit('/')[0:-1]) + '/', '&&', 'wget', '-c', url]
    try:
        output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
    except Exception:
        err += '\t'.join(command) + '\n'
        pass

    # [2] download unmapped bam
    url = base_url + '/%s/alignment/%s.unmapped*.bam' % (sample, sample)
    print(url)
    command = ['cd', '/'.join(log_path.rsplit('/')[0:-1]) + '/', '&&', 'wget', '-c', url]
    try:
        output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
    except Exception:
        err += '\t'.join(command) + '\n'
        pass

    url = base_url + '/%s/alignment/%s.mapped*.bam.bai' % (sample, sample)
    print(url)
    command = ['cd', '/'.join(log_path.rsplit('/')[0:-1]) + '/', '&&', 'wget', '-c', url]
    # [3] download mapped index
    try:
        output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
    except Exception:
        err += '\t'.join(command) + '\n'
        pass

    url = base_url + '/%s/alignment/%s.mapped*.bam' % (sample, sample)
    print(url)
    command = ['cd', '/'.join(log_path.rsplit('/')[0:-1]) + '/', '&&', 'wget', '-c', url]
    # [4] download mapped bam
    try:
        output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
    except Exception:
        err += '\t'.join(command) + '\n'
        pass
    print('wget section completed')

    # [5] rename the files
    mapped_bam = log_path+'%s.mapped.bam'%sample
    if not os.path.exists(mapped_bam):
        print('renamed mapped file not found, renaming...')
        command = ['mv',log_path+'/%s.mapped*.bam'%sample,mapped_bam]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass
        command = ['mv',log_path+'/%s.mapped*.bam.bai'%sample,mapped_bam+'.bai']
        try:
            output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    unmapped_bam = log_path+'%s.unmapped.bam'%sample
    if not os.path.exists(unmapped_bam):
        print('renamed unmapped file not found, renaming...')
        command = ['mv',log_path+'/%s.unmapped*.bam'%sample,unmapped_bam]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass
        command = ['mv',log_path+'/%s.unmapped*.bam.bai'%sample,unmapped_bam+'.bai']
        try:
            output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [6] merge the file
    merged_bam = log_path+'%s.merged.bam'%sample
    if not os.path.exists(merged_bam):
        print('merged mapped+unmapped bam file not found, merging...')
        command = ['sambamba','merge','-t',threads,'-l','9',merged_bam,mapped_bam,unmapped_bam]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [7] sort by name
    name_bam = log_path+'/%s.namesorted.bam'%sample
    if not os.path.exists(name_bam):
        print('name sorted merged bam file not found, sorting by name...')
        command = ['sambamba','sort','-n','-t',threads,'-m',memory+'GB','-l','9','-o',name_bam,merged_bam]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [8] bam to fastq format
    fq1,fq2 = log_path+'/%s_1.fq'%sample,log_path+'/%s_2.fq'%sample
    if not os.path.exists(fq1) or not os.path.exists(fq2):
        print('fastq format reads were not found, converting namesorted bam to fastq...')
        command = ['samtools','fastq','-@',threads,'-1',fq1,'-2',fq2,
                   '-0','/dev/null','-s','/dev/null','-n','-F','0x900',log_path+'/%s.namesorted.bam']
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [9] minimap2 realignment
    realigned_bam = log_path+'/%s.realigned.bam'%sample
    if not os.path.exists(realigned_bam):
        print('realigned bam was not found for sample %s, starting minimap2 with reference index %s'%(sample,ref_mmi))
        command = ['minimap2','-ax','sr','-t',threads,'-Y',ref_mmi,fq1,fq2,'|',
                   'samtools','view','-','-Sb','>',realigned_bam]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [10] sort the realigned bam
    sorted_realigned_bam = log_path+'/%s.realigned.sorted.bam'%sample
    if not os.path.exists(sorted_realigned_bam):
        print('sorted and realigned bam not found, starting sambamb sort on the realigned bam...')
        command = ['sambamba','sort','-t',threads,'-m',memory+'GB','-l','9','-o',sorted_realigned_bam,realigned_bam]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [11] mark duplicates
    final_bam = log_path+'/%s.final.bam'%sample
    if not os.path.exists(final_bam):
        print('final bam was not found, running duplicate marking...')
        command = ['sambamba','markdup','-r','-t',threads,'-l','9','--tmpdir',log_path,sorted_realigned_bam,final_bam]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [12] run hfm workflow
    hfm_file = log_path+'/%s.merged.hdf5'%sample
    if not os.path.exists(hfm_file):
        print('hfm file was not located, running hfm summary...')
        chroms = ['chr%s'%i for i in range(1,23,1)]+['chrX','chrY','chrM']
        command = ['extractor.py','-i',final_bam,'-o',log_path,'-w','100','-b','10','-p',threads,'-s',','.join(chroms)]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass

    # [13] rsync to large nfs disk

    # [14] verify on remote nfs disk

    # [15] clean up

    # [3a]execute the command here----------------------------------------------------
    # [3b]do a os directory/data check or a ls type command to check the size
    # [3b]of the produced data to ensure that everything is fine...
    if err == '':
        with open(log_path + '_' + sample, 'w') as f:
            f.write(output)
        return output
    else:
        with open(log_path + '_err_' + sample, 'w') as f:
            f.write(output+'\n'+err)
        return 'error on sample %s' % sample

def wget_fastq_align(base_url,log_path,ref_mmi,sample):
    print('starting sample %s'%sample)
    sample_dir = log_path+'/%s'%sample
    if os.path.exists(sample_dir): os.mkdir(sample_dir)
    output, err = '', ''

    # [1] download the fastq reads
    rgs = []
    for rg_tag in RG[sample]:
        rg = rg_tag.split('ID:')[-1].split('\t')[0]
        rgs += [rg]
        url = base_url+'/%s/sequence_read/%s_*.filt.fastq.gz'%(sample,rg)
        print(url)
        command = ['cd',sample_dir,'&&','wget','-c', url]
        try:
            output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass
    # [2] parse all of the finished file names, RG tags are already built in RG structure...
    fastqs = glob.glob(sample_dir+'/'+'.filt.fastq.gz')

    ~ / software / bioinfo / minimap2 - ax
    sr - Y - t
    12 - R
    '@RG\tID:SRR062641\tSM:HG00096\tPL:ILLUMINA\tPU:SRR062641\tLB:SOLEXA_SRR062641'
    ref / grch38_no_alt / grch38_no_alt.mmi
    SRR062641_1.filt.fastq.gz
    SRR062641_2.filt.fastq.gz | samtools
    view - -Sbh > HG00096.SRR062641.bam

    for fastq in fastq:
        command = ['minimap2','-ax','sr','-Y','-t',threads,'-R',rgs,ref_mmi,fq1,fq2,'|',
                   'samtools','view','-','-Sbh','>',rg1.bam]

    # [2] align each of them with the proper RG tags

    # [3] merge/sort them

    # [4] mark duplicates

    if err == '':
        with open(log_path+'_'+sample,'w') as f: f.write(output)
        return output
    else:
        with open(log_path+'_err_'+sample,'w') as f: f.write(output+'\n'+err)
        return 'error on sample %s' % sample

def wget_bas(base_url,log_path,sample):
    print('starting sample %s' % sample)
    output, err = '', ''
    # [1] download unmapped index
    url = base_url + '/%s/alignment/%s.mapped*.bam.bas'%(sample,sample)
    print(url)
    command = ['cd', '/'.join(log_path.rsplit('/')[0:-1]) + '/', '&&', 'wget', '-c', url]
    try:
        output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
    except Exception:
        err += '\t'.join(command) + '\n'
        pass
    if err == '':
        with open(log_path+'_'+sample,'w') as f: f.write(output)
        return output
    else:
        with open(log_path+'_err_'+sample,'w') as f: f.write(output+'\n'+err)
        return 'error on sample %s' % sample

results = []
def collect_results(result):
    results.append(result)

if __name__ == '__main__':
    base_url = 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/'
    log_path = out_dir
    print('using sample_list: %s' % sample_list_path)
    print('using sample ftp: %s' % sample_ftp_path)
    print('writing logs to %s' % log_path)
    print('checking the reference fasta file indexes...')
    ref_base,ref_dir = ref.rsplit('/')[-1].rsplit('.fa')[0],'/'.join(ref.rsplit('/')[:-1])
    ref_mmi = ref_dir+'/%s.mmi'%ref_base
    if not os.path.exists(ref_dir+'/%s.mmi'%ref_base):
        print('minimap2 index file being generated...')
        command = ['minimap2','-d',ref_mmi,ref]
        try:
            output = subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
        except Exception:
            err    = '\t'.join(command) + '\n'
            pass
    print('using %s threads for each of the %s connections with memory allocated at %sGB'%(threads,cpus,memory))
    P, N = {}, {}
    if sample_list != []: sample_list = []
    if sample_ftp != []: sample_ftp = []
    with open(sample_list_path, 'r') as f:
        sample_list = f.readlines()
    sample_list = [s.replace('\n', '').split('\t') for s in sample_list]
    with open(sample_ftp_path, 'r') as f:
        sample_ftp = f.readlines()
    sample_ftp = [s.replace('\n', '') for s in sample_ftp]

    for sample in sample_list:
        if sample[0] in sample_ftp:
            if P.has_key(sample[1]):
                P[sample[1]] += [sample[0]]
            else:
                P[sample[1]] = [sample[0]]
        else:
            if N.has_key(sample[1]):
                N[sample[1]] += [sample[0]]
            else:
                N[sample[1]] = [sample[0]]

    # ------------------------------------------------------
    pops = list(np.random.choice(P.keys(), num_samples, replace=True))
    pick_list = list(np.random.choice(list(set([y for k in P for y in P[k]])), num_samples, replace=False))
    # start || wget calls
    p1 = mp.Pool(processes=cpus)
    for sample in pick_list:  # for each sample download both mapped and unmapped patterns
        p1.apply_async(wget_bas,args=(base_url,log_path,sample),callback=collect_results)
        time.sleep(1)
    p1.close()
    p1.join()

    L = []
    for i in results:
        if not i.startswith('error on sample'): L += [i]
        else: print(i)
    print('%s samples were successfully downloaded' % len(L))
