#!/usr/bin/env python
import argparse
import time
import datetime
import sys
import os
import glob
import numpy as np
import multiprocessing as mp
import subprocess32 as subprocess

des = """1000 genomes low coverage ftp download management and fastq alignment to synchronized hfm bot"""
parser = argparse.ArgumentParser(description=des)
parser.add_argument('-l', '--pop_list', type=str, help='path to the tsv population read group file\t[None]')
parser.add_argument('-o', '--out_dir', type=str, help='output directory to save ...bam/ into\t[None]')
parser.add_argument('--target', type=str, help='hostname target for final hfm transfer\t[None]')
parser.add_argument('--software', type=str, help='software path for tools\t[~/software/bioinfo/]')
parser.add_argument('--target_dir', type=str, help='target directory for final hfm transfer\t[None]')
parser.add_argument('-r', '--reference', type=str, help='reference fasta file\t[None]')
parser.add_argument('-s', '--seqs', type=str, help='comma seperated list of seqs to process in hfm\t[ALL]')
parser.add_argument('-p','--select_by_pop',action='store_true',help='select from 26 pop types versus the max coverage\t[False]')
parser.add_argument('-b', '--branch', type=int, help='hfm window branching factor\t[10]')
parser.add_argument('-w', '--window', type=int, help='hfm summary window size in bp\t[100]')
parser.add_argument('--no_merge_rg', action='store_true', help='do not merge the rg during hfm generation\t[False]')
parser.add_argument('-c', '--connections', type=int, help='number of connections to use at once\t[1]')
parser.add_argument('-t', '--threads', type=int, help='number of threads per connection\t[1]')
parser.add_argument('-m', '--memory', type=int, help='total GB to allocate per connection   \t[1]')
parser.add_argument('-n', '--num_samples', type=str, help='total number of samples to download\t[1]')
parser.add_argument('--high_cov', action='store_true', help='get the high coverage data\t[False]')
args = parser.parse_args()

if args.pop_list is not None:
    sample_list_path = args.pop_list
else:
    raise IOError
if args.out_dir is not None:
    out_dir = args.out_dir
    if not os.path.exists(out_dir): os.makedirs(out_dir)
else:
    raise IOError
if args.target is not None:
    target = args.target
else:
    raise IOError
if args.target_dir is not None:
    target_dir = args.target_dir
else:
    raise IOError
if args.software is not None:
    software = args.software
else:
    software = '~/software/bioinfo/'
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
if args.window is not None:
    window = args.window
else:
    window = 100
if args.branch is not None:
    branch = args.branch
else:
    branch = 10
if args.no_merge_rg:
    merge_rg = False
else:
    merge_rg = True
if args.num_samples is not None and args.num_samples.upper() != 'ALL':
    num_samples = int(args.num_samples)
elif args.num_samples.upper() == 'ALL':
    sample_list = []
    with open(sample_list_path, 'r') as f:
        sample_list = [s.replace('\n', '').split('\t') for s in f.readlines()]
    P,RG = {},{}
    for sample in sample_list:
        if P.has_key(sample[1]):
            P[sample[1]] += [sample[0]]
        else:
            P[sample[1]] = [sample[0]]
        RG[sample[0]] = ['@RG\tID:%s\tSM:%s\tPL:%s\tLB:%s\tPU:%s'%\
                         (rg.split(';')[0],sample[0],rg.split(';')[2],rg.split(';')[1],rg.split(';')[0]) for rg in sample[2].split(',')]
    num_samples = sum([len(P[k]) for k in P])
else:
    sample_list = []
    num_samples = 1

#write stage status and time on the logfile
#key: stage=11\tstatus=0 means started, status=1 means finished successfully, status=-1 means failed
def write_log_status(log_path,stage,status):
    with open(log_path,'a') as f:
        dt = datetime.datetime.now().strftime('%Y/%m/%d[%H:%M:%S]')
        f.write('stage=%s\tstatus=%s\tdatetime=%s\n'%(stage,status,dt))
        return True
    return False

#will read all of the stages and status lines and check agaist the stages that should be completed
#stage 1 is the start of the jobs, while stage 100 is the code for all stages completed successfully
#key stage=1
def read_log_status(log_path):
    L = {}
    with open(log_path,'r') as f:
        raw = [row.replace('\n','') for row in f.readlines()]
        if len(raw)>0:  #more than one entry means we either started or finished a stage>=1 and stage<=100
            while raw[-1]=='': raw = raw[:-1] #trim extra lines...
            for i in range(len(raw)):
                stage  = int(raw[i].rsplit('stage=')[-1].rsplit('\t')[0])
                status = int(raw[i].rsplit('status=')[-1].rsplit('\t')[0])
                dt     = datetime.datetime.strptime(raw[i].rsplit('datetime=')[-1].rsplit('\t')[0],'%Y/%m/%d[%H:%M:%S]')
                if stage in L: L[stage] += [[status,dt]]
                else:          L[stage]  = [[status,dt]]
    return L

#given the analysis of the last successful stage and the old file
#delete the appropriate number of lines after a restart (IE clear the last unsuccessful stage)
#we assume that the failure was not do to issues with the script (IE it was perfect)
def restart_log_status(log_path,stage):
    s = ''
    with open(log_path,'r') as f: raw = [row.replace('\n','') for row in f.readlines()]
    for row in raw:
        if not row.startswith('stage=%s'%stage): s += row+'\n'
    if s!='':
        with open(log_path,'w') as f:
            f.write(s)
            return True
    return False

#go through the stages and find the last success
#verbose will tell you how many hours have elapsed in total on the job
def get_stage_location(L):
    last_id = 0 #no stages have been completed...
    for l in sorted(L.keys()):
        if len(L[l])>1:
            for j in range(len(L[l])):
                if L[l][j][0]==1:
                    last_id = l
    return last_id

def associate_bas_rgs(bas_dir):
    B,bas = {},sorted(glob.glob(bas_dir+'/*.bas'))
    for b in bas:
        with open(b,'r') as f: data = [i.replace('\n','').split('\t') for i in f.readlines()]
        if len(data)>0: #header should be: 0:'bam_filename', 1:'md5', 2:'study', 3:'sample', 4:'platform', 5:'library', 6:'readgroup',...
            for i in range(1,len(data),1):
                sample,platform,library,readgroup = data[i][3],data[i][4],data[i][5],data[i][6]
                cov,mapq = (int(data[i][8])*1.0)/3E9,float(data[i][14])
                if sample in B: B[sample] += [[readgroup,library,platform,cov,mapq]]
                else:           B[sample]  = [[readgroup,library,platform,cov,mapq]]
    return B

# main workflow for downloading fastq files and producing alignment files for a reference with duplicate marked entries for hfm creation
def wget_fastq_align(base_url,log_path,ref_mmi,sample,merge_rg):
    #preliminary:::::::::::::::::::::::::::::::::::::::
    print('starting sample %s'%sample)
    sample_dir = log_path+'/%s'%sample
    log_status = log_path+'/%s_status.log'%sample
    if os.path.exists(log_status):
        print('reading log status for %s'%sample)
        L = read_log_status(log_status)
        last_id = get_stage_location(L)
    else:
        last_id = 0
    print('last stage completed for %s was stage=%s'%(sample,last_id))
    if last_id>99: return 'job is already complete'
    elif not os.path.exists(sample_dir): os.mkdir(sample_dir)
    print('finished preliminary logging structures for sample %s'%sample)
    #::::::::::::::::::::::::::::::::::::::::::::::::::

    if last_id==0: # [1] download the fastq reads
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        #-----------------------------------
        rgs = []
        for rg_tag in RG[sample]:
            rg = rg_tag.split('ID:')[-1].split('\t')[0]
            rgs += [rg]
            url = base_url+'/%s/sequence_read/%s_*.filt.fastq.gz'%(sample,rg)
            print(url)
            command = ['cd',sample_dir,'&&','wget','-c',url] #can restart downloads with -c and a failure
            try:
                output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
            except Exception:
                err += ' '.join(command) + '\n'
                pass
        #------------------------------------------------------
        if err=='' and len(glob.glob(sample_dir+'/'+'*.filt.fastq.gz'))>=len(RG[sample]): #at least
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample %s stage %s: %s :%s'%(sample,stage,err,output)
        #------------------------------------------------------
    if last_id==1: # [2] parse all of the finished file names, RG tags are in the RG structure
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        #-----------------------------------
        fastqs = glob.glob(sample_dir+'/'+'*.filt.fastq.gz') #set up reads to readgroup associations
        fqs = {rg.split('ID:')[-1].split('\t')[0]:{} for rg in RG[sample]}
        for fastq in fastqs:
            idx,read = fastq.rsplit('/')[-1].rsplit('_')[0],fastq.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0]
            if idx in fqs: fqs[idx][read] = fastq
        rgs,F = fqs.keys(),{}
        for rg in rgs:
            if len(fqs[rg])<2:fqs.pop(rg)
        if len(fqs)<1: err += 'error occured with files %s\nprocessed as %s'%(fastqs,fqs)

        #now we have figured out which files we have and can further process via minimap2 pipeline
        print('starting minimap2 checks for sample: %s'%sample)
        for idx in fqs:
            rg = ''
            for r in RG[sample]:
                if r.find(idx)>-1: rg = r
            print("checking minimap2 alignment of read group = '%s'"%rg.replace('\t','\\t'))
            check_bam = glob.glob('%s/%s_%s.bam'%(sample_dir,sample,idx))
            print('found minimap2 alignment of read group = %s'%(len(check_bam)>0))
            if rg!='' and len(check_bam)<1:
                print("starting new minimap2 alignment for read group = '%s'"%rg.replace('\t','\\t'))
                command = [software+'minimap2','-ax','sr','-Y','-t %s'%threads,'-R',"'%s'"%rg.replace('\t','\\t'),ref_mmi,fqs[idx]['1'],fqs[idx]['2'],'|',
                           software+'samtools','view','-','-Sbh','>','%s/%s_%s.bam'%(sample_dir,sample,idx)]
                try:
                    output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
                    print("finished minimap2 alignemnt of read group = '%s'"%rg)
                except Exception as E:
                    print("minimap2 alignemnt failed for read group = '%s'"%rg)
                    err += 'err '+' '.join(command)+' E '+str(E)+'\n'
                    pass
            else:
                if rg=='': err += 'sample: %s rg was null'%sample
                if len(check_bam)>0: err += 'sample: %s rg: %s was found on disk\n'%(sample,rg)
        print('completed checks for minmap2 for sample: %s'%sample)
        #------------------------------------------------------
        finished_bams = glob.glob(sample_dir+'/'+'%s_*.bam'%sample)
        missing_bams = []
        #locate the missing RG[sample] in the bams...?
        for rg in RG[sample]:
            r = rg.split('ID:')[-1].split('\t')[0]
            if not any([bams.find(r)>0 for bams in finished_bams]):
                missing_bams += [r]
        # locate the missing RG[sample] in the bams...?
        if len(finished_bams)>=len(RG[sample]): #at least all passed
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample: %s\tstage: %s\tRGs: %s/%s missing=%s\nerr: %s\nout:%s'%\
                   (sample,stage,len(finished_bams),len(RG[sample]),missing_bams,err,output)
        #------------------------------------------------------
    if last_id==2: # [3] need to coordinate sort the seperate read now
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        #-----------------------------------
        sample_bams = glob.glob(sample_dir+'/%s_*.bam'%sample)
        for bam in sample_bams:
            command = [software+'sambamba','sort','-t %s'%threads,'-m %sGB'%memory,'--tmpdir=%s'%sample_dir,'-l','9',bam]
            try:
                output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
            except Exception:
                err += ' '.join(command) + '\n'
                pass
        if err=='' and len(glob.glob(sample_dir+'/%s_*.sorted.bam'%sample))>=len(RG[sample]): #at least
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample %s stage %s: %s :%s'%(sample,stage,err,output)
        #------------------------------------------------------
    if last_id==3: #[4] merge bam files
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        #-----------------------------------
        sorted_sample_bams = glob.glob(sample_dir+'/%s_*.sorted.bam'%sample)
        if len(sorted_sample_bams)>0 and len(sorted_sample_bams)<=1:
            command = ['mv',sorted_sample_bams[0],sample_dir+'/%s.merged.bam'%sample] #upgrade the name
            try:
                output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
            except Exception:
                err += ' '.join(command) + '\n'
                pass
        elif len(sorted_sample_bams)>1:
            command = [software+'sambamba','merge','-t %s'%threads,'-l','9',sample_dir+'/%s.merged.bam'%sample]+sorted_sample_bams
            try:
                output += subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
            except Exception:
                err += ' '.join(command) + '\n'
                pass
        if err=='' and len(glob.glob(sample_dir+'/%s.merged.bam'%sample))>=1: #at least
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample %s stage %s: %s :%s'%(sample,stage,err,output)
    if last_id==4: # [5] mark duplicates
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        # -----------------------------------
        command = [software+'sambamba','markdup','-t %s'%threads,'--tmpdir=%s'%sample_dir,'-l','9',
                   sample_dir+'/%s.merged.bam'%sample,sample_dir+'/%s.final.bam'%sample]
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += ' '.join(command) + '\n'
            pass
        if err=='' and len(glob.glob(sample_dir+'/%s.final.bam'%sample))>=1: #at least
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample %s stage %s: %s :%s'%(sample,stage,err,output)
    if last_id==5: # [6] run hfm on the final BAM file
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        # -----------------------------------
        if merge_rg:
            command = [software+'extractor.py','-i',sample_dir+'/%s.final.bam'%sample,'-o',sample_dir,
                       '-s',"'%s'"%','.join(sorted(all_seqs.keys())),'-w %s'%window,'-b %s'%branch,'-p %s'%threads]
        else:
            command = [software+'extractor.py','-i',sample_dir+'/%s.final.bam'%sample,'-o',sample_dir,'--no_merge_rg',
                       '-s',"'%s'"%','.join(sorted(all_seqs.keys())),'-w %s'%window,'-b %s'%branch,'-p %s'%threads]
        print(' '.join(command))
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += ' '.join(command) + '\n'
            pass
        if err=='' and len(glob.glob(sample_dir+'/%s.final.merged.hdf5'%sample))>=1: #at least
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample %s stage %s: %s :%s'%(sample,stage,err,output)
    if last_id==6: # [7] sync the data to larger disk if needed
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        # -----------------------------------
        hdf5_file = sample_dir+'/%s.final.merged.hdf5'%sample
        dest = '%s:%s'%(target,target_dir)
        command = ['rsync','-aP',hdf5_file,dest]
        print(' '.join(command))
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += ' '.join(command) + '\n'
            pass
        if err=='': #at least
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample %s stage %s: %s :%s'%(sample,stage,err,output)
    if last_id==7: # [8] clean up local disks if needed then shutdown
        stage,output,err = last_id+1,'',''
        write_log_status(log_status,stage,0)
        # -----------------------------------
        command = ['rm','-rf',sample_dir] #test it first
        try:
            output += subprocess.check_output(' '.join(command),stderr=subprocess.STDOUT,shell=True)
        except Exception:
            err += '\t'.join(command) + '\n'
            pass
        if err=='': #at least
            write_log_status(log_status,stage,1)
            last_id+=1 #continue
        else:
            write_log_status(log_status,stage,-1)
            return 'error on sample %s stage %s: %s :%s'%(sample,stage,err,output)
    if last_id==8: #log the final terminal stage
        write_log_status(log_status,100,0)
        write_log_status(log_status,100,1)
        return 'sample %s finished processing from ftp to hfm'%sample

#tool for grabbing the bas files
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

def get_seqs_lens(ref_dict):
    seqs,raw = {},[]
    with open(ref_dict,'r') as f:
        raw = [row.replace('\n','') for row in f.readlines()]
    for row in raw:
        if row.startswith('@SQ'):
            seqs[row.rsplit('SN:')[-1].rsplit('\t')[0]] = row.rsplit('LN:')[-1].rsplit('\t')[0]
    return seqs

results = []
def collect_results(result):
    results.append(result)

if __name__ == '__main__':
    base_url = 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/'
    log_path = out_dir
    print('using sample_list: %s' % sample_list_path)
    print('writing logs to %s' % log_path)
    print('checking the reference fasta file indexes...')
    ref_base,ref_dir = ref.rsplit('/')[-1].rsplit('.fa')[0],'/'.join(ref.rsplit('/')[:-1])
    ref_mmi = ref_dir+'/%s.mmi'%ref_base
    all_seqs = get_seqs_lens(ref_dir+'/%s.dict'%ref_base)
    if args.seqs is not None:
        SKS,seqs = {},args.seqs.rsplit(',')
        valid = True
        for s in seqs:
            if s not in all_seqs:
                valid = False
            else:
                SKS[s] = all_seqs[s]
        if valid: all_seqs = SKS
    print('using sequences=%s'%sorted(all_seqs))
    if not os.path.exists(ref_mmi):
        print('minimap2 index file being generated...')
        command = [software+'minimap2','-d',ref_mmi,ref]
        try:
            output = subprocess.check_output(' '.join(command), stderr=subprocess.STDOUT, shell=True)
        except Exception:
            err    = '\t'.join(command) + '\n'
            pass
    print('using %s threads for each of the %s connections with memory allocated at %sGB'%(threads,cpus,memory))
    P,RG,C = {},{},{}
    sample_list = []
    with open(sample_list_path, 'r') as f:
        sample_list = [s.replace('\n', '').split('\t') for s in f.readlines()]
    for sample in sample_list:
        if P.has_key(sample[1]):
            P[sample[1]] += [sample[0]]
        else:
            P[sample[1]] = [sample[0]]
        RG[sample[0]] = ['@RG\tID:%s\tSM:%s\tPL:%s\tLB:%s\tPU:%s'%\
                         (rg.split(';')[0],sample[0],rg.split(';')[2],rg.split(';')[1],rg.split(';')[0]) \
                         for rg in sample[2].split(',')]
        C[sample[0]] = sum([float(rg.split(';')[3]) for rg in sample[2].split(',')])
    # ------------------------------------------------------------------------------------------------------------------
    if args.select_by_pop: #select evenly among pop types
        print('selecting downloads by population')
        pops = list(np.random.choice(P.keys(), num_samples, replace=True))
        pick_list = list(np.random.choice(list(set([y for k in P for y in P[k]])), num_samples, replace=False))
    else:                 #select the highest coverage samples using the bas RG data
        print('using highest %sth coverage bas for sample selection'%num_samples)
        pick_list = sorted(C,key=lambda x: C[x])[::-1][:num_samples] #side effect of downloading larger first
    # start || wget calls
    p1 = mp.Pool(processes=cpus)
    for sample in pick_list:  # for each sample download both mapped and unmapped patterns
        p1.apply_async(wget_fastq_align,args=(base_url,log_path,ref_mmi,sample,merge_rg),callback=collect_results)
        time.sleep(1)
    p1.close()
    p1.join()

    L = []
    for i in results:
        if not i.startswith('error on sample'): L += [i]
        else: print(i)
    print('%s samples were successfully downloaded' % len(L))
