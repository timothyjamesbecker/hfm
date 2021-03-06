#hfm/core.pyx
#Copyright (C) 2019-2020 Timothy James Becker

#c imports
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from cpython cimport PyBytes_FromStringAndSize
from pysam.libcfaidx cimport FastaFile
from pysam.libcalignmentfile cimport AlignmentFile, IteratorRowRegion
from pysam.libcalignedsegment cimport AlignedSegment
cimport cython
cimport numpy as np
#regular imports
import math
import numpy as np
import mappy
__version__ = '0.1.8'

#feature defines
cdef unsigned int N,SUM,MIN,MAX,M1,M2,M3,M4,FN
N,SUM,MIN,MAX,M1,M2,M3,M4,FN = 0,1,2,3,4,5,6,7,8

#type defines so we can change the data sizes?
#:::TO DO:::----------------------------------
ctypedef float  minp_t;
ctypedef double mout_t;
ctypedef float sinp_t;
ctypedef float sout_t;
ctypedef float tinp_t;
ctypedef float tout_t;
#:::TO DO:::----------------------------------

#this one is without a GIL and assumes that |c1|>=|c2|
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def edit_dist(const unsigned char[::1] c1, const unsigned char[::1] c2,
              unsigned int w_mat=0, unsigned int w_ins=1, unsigned int w_del=1, unsigned int w_sub=1):
    cdef unsigned int i,j,k,u,v,x,y,z
    u,v,k = len(c1),len(c2),2
    cdef np.ndarray[unsigned int, ndim=2] D = np.zeros([u+1,k], dtype=np.uint32)
    for i in range(u+1):   D[i][0] = i
    for j in range(v%k+1): D[0][j] = j
    for j in range(1,v+1):
        for i in range(1,u+1):
            if c1[i-1] == c2[j-1]:
                D[i][j%k] = D[i-1][(j-1)%k]+w_mat  #matching
            else:                                  #mismatch, del, ins, sub
                x,y,z = D[i-1][j%k]+w_del,D[i][(j-1)%k]+w_ins,D[i-1][(j-1)%k]+w_sub
                if x<=y and x<=z:   D[i][j%k] = x
                elif y<=x and y<=z: D[i][j%k] = y
                else:               D[i][j%k] = z
    return D[u][v%k]

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def edit_dist_str(str s1, str s2, list w=[1,1,1]):
    cdef unsigned int i,j,k,u,v
    u,v,k = len(s1),len(s2),2
    if u<v: u,v,s1,s2 = v,u,s2,s1 #flip to longest of the two
    cdef np.ndarray[unsigned int, ndim=2] D = np.zeros([u+1,k], dtype=np.uint32)
    for i in range(u+1):   D[i][0] = i
    for j in range(v%k+1): D[0][j] = j
    for j in range(1,v+1):
        for i in range(1,u+1):
            if s1[i-1] == s2[j-1]:
                D[i][j%k] = D[i-1][(j-1)%k] #matching
            else:                           #mismatch, del, ins, sub
                D[i][j%k] = min(D[i-1][j%k]+w[0],D[i][(j-1)%k]+w[1],D[i-1][(j-1)%k]+w[2])
    return D[u][v%k]


#use this for slightly faster default feature constructions using numpy broadcasting...
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def load_reads_all_tracks(str alignment_path, dict sms, str seq, int start, int end,
                          select=['alternate','orient_same','orient_out','orient_um','orient_chr','splice',
                                  'clipped','right_clipped','left_clipped','big_del','deletion','insertion','substitution',
                                  'fwd_rev_diff','total','primary','proper_pair','discordant','mapq','mapq_pp','mapq_dis',
                                  'tlen_pp','tlen_dis','tlen_pp_rd','tlen_dis_rd','RD','GC','MD',
                                  'left_smap_same','left_smap_diff','right_smap_same','right_smap_diff'],
                          bint merge_rg=True, bint dna=False, bint exact_sub=False,
                          str ref_seq = None, str align_preset = 'sr',
                          int min_anchor=36, int min_clip=18, int min_smapq=20, int big_del=36,
                          mem_map_path=None):
    cdef int            i,j,a,b,c,d,e,f,s1,s2,tid,mid,mapq,alen,rlen,aend,rend,tlen,last,offset
    cdef float          x,y,z
    cdef str            k,t,v,rg,tag,track,sequence
    cdef list           tags,cigar,cigar_list,dnas
    cdef dict           C,S,A
    cdef AlignmentFile  samfile
    cdef AlignedSegment read
    C,S,rg = {},{},'all'
    tracks = ['alternate','orient_same','orient_out','orient_um','orient_chr','right_anchor','left_anchor','splice',
              'right_clipped','left_clipped','clipped','big_del','deletion','insertion','substitution','fwd_rev_diff']
    tracks = sorted(list(set(select).intersection(set(tracks))))
    values = ['total','primary','proper_pair','discordant','mapq','mapq_pp','mapq_dis',
              'tlen','tlen_pp','tlen_dis','tlen_pp_rd','tlen_dis_rd','orient_same_rd','orient_out_rd','RD','GC','MD',
              'smap_same','smap_diff','left_smap_same','left_smap_diff','right_smap_same','right_smap_diff']
    values = sorted(list(set(select).intersection(set(values))))
    dna_trans = ['A-A','A-C','A-G','A-T','C-A','C-C','C-G','C-T','G-A','G-C','G-G','G-T','T-A','T-C','T-G','T-T']
    if merge_rg: sms = {'all':'-'.join(sorted(list(set(sms.values()))))} #duplicated from the safe lib
    if mem_map_path is None:
        print('building in mem arrays for seq=%s'%seq)
        for rg in sms:
            for t in tracks:
                if t in C: C[t][rg] = np.zeros([end-start,], dtype=np.float32)
                else:      C[t] = {rg:np.zeros([end-start,], dtype=np.float32)}
            for v in values: #don't have to deal with 0-div
                if v in C: C[v][rg] = np.zeros([end-start,], dtype=np.float32)+1.0
                else:      C[v] = {rg:np.zeros([end-start,], dtype=np.float32)+1.0}
            if dna:
                for k in dna_trans:
                    if k in C: C[k][rg] = np.zeros([end-start], dtype=np.float32)
                    else:      C[k] = {rg:np.zeros([end-start], dtype=np.float32)}
            S[rg] = {'L':[],'R':[]}
        print('finished building in mem arrays for seq=%s'%seq)
    else:
        print('building mem map arrays for seq=%s'%seq)
        for rg in sms:
            for t in tracks:
                if t in C:
                    C[t][rg] = np.memmap(mem_map_path+seq+'_trk_'+t+'_rg_'+rg+'.dat', shape=(end-start,),dtype=np.float32,mode='w+')
                    print(type(C[t][rg]))
                else:
                    C[t] = {rg:np.memmap(mem_map_path+seq+'_trk_'+t+'_rg_'+rg+'.dat', shape=(end-start,),dtype=np.float32,mode='w+')}
            for v in values: #don't have to deal with 0-div
                if v in C:
                    C[v][rg] = np.memmap(mem_map_path+seq+'_trk_'+v+'_rg_'+rg+'.dat', shape=(end-start,),dtype=np.float32,mode='w+')
                    C[v][rg][:] += 1.0
                else:
                    C[v] = {rg:np.memmap(mem_map_path+seq+'_trk_'+v+'_rg_'+rg+'.dat', shape=(end-start,),dtype=np.float32,mode='w+')}
                    C[v][rg][:] += 1.0
            if dna:
                for k in dna_trans:
                    if k in C:
                        C[k][rg] = np.memmap(mem_map_path+seq+'_trk_'+k+'_rg_'+rg+'.dat', shape=(end-start,),dtype=np.float32,mode='w+')
                    else:
                        C[k] = {rg:np.memmap(mem_map_path+seq+'_trk_'+k+'_rg_'+rg+'.dat', shape=(end-start,),dtype=np.float32,mode='w+')}
            S[rg] = {'L':[],'R':[]}
        print('finished building the mem map arrays for seq=%s'%seq)
    samfile = AlignmentFile(alignment_path,'rb')
    for read in samfile.fetch(start=start,end=end,region=seq,until_eof=True):
        if read.reference_end is not None and not read.is_duplicate and not read.is_qcfail:
            pos   = read.pos
            aend  = read.reference_end
            alen  = aend-pos
            sequence = read.query_sequence.upper()
            rlen = min(read.rlen,len(sequence))
            rend = pos+rlen
            tid  = read.reference_id
            tlen = read.template_length
            mid  = read.next_reference_id
            mapq = read.mapping_quality
            cigar = read.cigartuples
            c = (start-pos if start-pos>0 else 0)   #amount of lefthand array  boundry overange
            d = (rend-end if rend-end>0 else 0)     #amount of righthand array boundry overage
            if pos<start:            a = 0          #clips to array start
            else:                    a = pos-start  #in between start and end
            if end-start<rend-start: b = end-start  #clips to array end
            else:                    b = rend-start #in between start and end
            if not merge_rg:                 #if read group tag merging is enabled, skip this part
                tags = read.get_tags()       #all optional tags
                for j in range(len(tags)):   #find the RG string in the read
                    if tags[j][0]=='RG':     #no matching to prexisting rg here...
                        rg = tags[j][1]
                        break
            if 'total' in C:                          C['total'][rg][a:b]        += 1.0
            if 'mapq' in C:                           C['mapq'][rg][a:b]         += mapq
            if 'tlen' in C:                           C['tlen'][rg][a:b]         += tlen
            if read.is_proper_pair:
                if 'proper_pair' in C:                C['proper_pair'][rg][a:b]  += 1.0
                if 'mapq_pp' in C:                    C['mapq_pp'][rg][a:b]      += mapq
                if 'tlen_pp' in C:                    C['tlen_pp'][rg][a:b]      += tlen
                if 'tlen_pp_rd' in C and tlen>0.0:    C['tlen_pp_rd'][rg][a:min(b+tlen,end-start)] += 1.0   #tlen projection across RD
                elif 'tlen_pp_rd' in C and tlen<0.0:  C['tlen_pp_rd'][rg][max(0,a+tlen):b]         += 1.0 #neg tlen projection across RD
            elif tid==mid and not read.mate_is_unmapped:
                if 'discordant' in C:                 C['discordant'][rg][a:b]   += 1.0
                if 'mapq_dis' in C:                   C['mapq_dis'][rg][a:b]     += mapq
                if 'tlen_dis' in C:                   C['tlen_dis'][rg][a:b]     += tlen
                if 'tlen_dis_rd' in C and tlen>0.0:   C['tlen_dis_rd'][rg][a:min(b+tlen,end-start)] += 1.0   #tlen projection across RD
                elif 'tlen_dis_rd' in C and tlen<0.0: C['tlen_dis_rd'][rg][max(0,a+tlen):b]         += 1.0 #neg tlen projection across RD
            if not read.is_supplementary and not read.is_secondary:
                if 'primary' in C:     C['primary'][rg][a:b]      += 1.0
            else:
                if 'alternate' in C:   C['alternate'][rg][a:b]    += 1.0
            if 'fwd_rev_diff' in C:
                if not read.is_reverse:C['fwd_rev_diff'][rg][a:b] += 1.0
                else:                  C['fwd_rev_diff'][rg][a:b] -= 1.0
            if ((read.is_reverse and read.mate_is_reverse) or\
                (not read.is_reverse and not read.mate_is_reverse)):
                if 'orient_same' in C:                       C['orient_same'][rg][a:b]  += 1.0
                if 'orient_same_rd' in C and tlen>0.0:       C['orient_same_rd'][rg][a:min(b+tlen,end-start)] += 1.0   #tlen projection across RD
                elif 'orient_same_rd' in C and tlen<0.0:     C['orient_same_rd'][rg][max(0,a+tlen):b]         += 1.0
            elif ((read.is_reverse and tlen>0) or\
                (not read.is_reverse and tlen <0)):
                if 'orient_out' in C:                        C['orient_out'][rg][a:b]   += 1.0
                if 'orient_out_rd' in C and tlen>0.0:        C['orient_out_rd'][rg][a:min(b+tlen,end-start)] += 1.0   #tlen projection across RD
                elif 'orient_out_rd' in C and tlen<0.0:      C['orient_out_rd'][rg][max(0,a+tlen):b]         += 1.0
            elif 'orient_um' in C and read.mate_is_unmapped: C['orient_um'][rg][a:b]    += 1.0
            elif 'orient_chr' in C and tid!=mid:             C['orient_chr'][rg][a:b]   += 1.0
            if len(sequence)>0:
                C['GC'][rg][a:b]  += <float>(sequence.count('G')+sequence.count('C'))/<float>(len(sequence))
                if dna: #dna transistion feature calculations
                    C['A-A'][rg][a:b]  += <float>(sequence.count('AA'))/<float>(len(sequence))
                    C['A-C'][rg][a:b]  += <float>(sequence.count('AC'))/<float>(len(sequence))
                    C['A-G'][rg][a:b]  += <float>(sequence.count('AG'))/<float>(len(sequence))
                    C['A-T'][rg][a:b]  += <float>(sequence.count('AT'))/<float>(len(sequence))
                    C['C-A'][rg][a:b]  += <float>(sequence.count('CA'))/<float>(len(sequence))
                    C['C-C'][rg][a:b]  += <float>(sequence.count('CC'))/<float>(len(sequence))
                    C['C-G'][rg][a:b]  += <float>(sequence.count('CG'))/<float>(len(sequence))
                    C['C-T'][rg][a:b]  += <float>(sequence.count('CT'))/<float>(len(sequence))
                    C['G-A'][rg][a:b]  += <float>(sequence.count('GA'))/<float>(len(sequence))
                    C['G-C'][rg][a:b]  += <float>(sequence.count('GC'))/<float>(len(sequence))
                    C['G-G'][rg][a:b]  += <float>(sequence.count('GG'))/<float>(len(sequence))
                    C['G-T'][rg][a:b]  += <float>(sequence.count('GT'))/<float>(len(sequence))
                    C['T-A'][rg][a:b]  += <float>(sequence.count('TA'))/<float>(len(sequence))
                    C['T-C'][rg][a:b]  += <float>(sequence.count('TC'))/<float>(len(sequence))
                    C['T-G'][rg][a:b]  += <float>(sequence.count('TG'))/<float>(len(sequence))
                    C['T-T'][rg][a:b]  += <float>(sequence.count('TT'))/<float>(len(sequence))
            #cigar ops based calculations---------------------
            if cigar is not None:
                cigar_list,j = [],0
                for i in range(len(cigar)):
                    if cigar[i][0]!=1 and cigar[i][0]!=6:  #insert and pad do not move in the reference space
                        e = min(j+cigar[i][1],rlen-d)        #saturates to right side sequence length
                        if e>c:
                            if j>c: cigar_list += [(cigar[i][0],j,e)]
                            else:   cigar_list += [(cigar[i][0],c,e)]
                        j = e
                    else: cigar_list += [(cigar[i][0],j,j)]
                if len(cigar_list)>0:
                    if cigar_list[0][0]==4 or cigar_list[0][0]==5: #'S' or 'H'
                        e = min(a+cigar_list[0][1],b)
                        f = min(a+cigar_list[0][2],b)
                        if f-e>=min_clip:
                            if 'clipped' in C:          C['clipped'][rg][e:f]       += 1.0
                            if 'left_clipped' in C:     C['left_clipped'][rg][e:f]  += 1.0
                            S[rg]['L'] += [[sequence[:cigar[0][1]],cigar,
                                            pos+cigar[0][1],tlen,(-1 if read.is_reverse else 1)]]
                        if cigar_list[len(cigar_list)-1][0]==0 or cigar_list[len(cigar_list)-1][0]==7: #'M' or '='
                            if f-e>=min_anchor:
                                if 'right_anchor' in C: C['right_anchor'][rg][e:f]  += 1.0
                    if cigar_list[len(cigar_list)-1][0]==4 or cigar_list[len(cigar_list)-1][0]==5: #'S' or 'H'
                        e = min(a+cigar_list[len(cigar_list)-1][1],b)
                        f = min(a+cigar_list[len(cigar_list)-1][2],b)
                        if f-e>=min_clip:
                            if 'clipped' in C:          C['clipped'][rg][e:f]       += 1.0
                            if 'right_clipped' in C:    C['right_clipped'][rg][e:f] += 1.0
                            S[rg]['R'] += [[sequence[cigar[len(cigar_list)-1][1]:],cigar,
                                            pos-cigar[len(cigar_list)-1][1],tlen,(-1 if read.is_reverse else 1)]]
                        if cigar_list[0][0]==0 or cigar_list[0][0]==7: #'M'
                            if f-e>=min_anchor:
                                if 'left_anchor' in C:  C['left_anchor'][rg][e:f]   += 1.0
                    if ref_seq is not None and exact_sub:
                        for i in range(len(cigar_list)):
                            e = min(a+cigar_list[i][1],b)
                            f = min(a+cigar_list[i][2],b)
                            if cigar_list[i][0]==1:
                                if 'insertion' in C:            C['insertion'][rg][e-1:f]    += 1.0  #single coordinates
                            if cigar_list[i][0]==2:
                                if 'deletion' in C:             C['deletion'][rg][e:f]       += 1.0
                                if f-e>=big_del:
                                    if 'big_del' in C:          C['big_del'][rg][e:f]        += 1.0
                            if cigar_list[i][0]==3:
                                if 'splice' in C:               C['splice'][rg][e:f]         += 1.0
                            if cigar_list[i][0]==0 or cigar_list[i][0]==7: #check matches for subs
                                for j in range(cigar_list[i][1],cigar_list[i][2],1):
                                    if sequence[j]!=ref_seq[pos+j]:
                                        if 'substitution' in C: C['substitution'][rg][a+j-c] += 1.0
                    else: #this is the more acurate exact matching algorithm-----------------------------
                        for i in range(len(cigar_list)):
                            e = min(a+cigar_list[i][1],b)
                            f = min(a+cigar_list[i][2],b)
                            if cigar_list[i][0]==1:
                                if 'insertion' in C:            C['insertion'][rg][e-1:f]    += 1.0  #single coordinates
                            if cigar_list[i][0]==2:
                                C['deletion'][rg][e:f] += 1.0
                                if f-e>=big_del:
                                    if 'big_del' in C:          C['big_del'][rg][e:f]        += 1.0
                            if cigar_list[i][0]==8:
                                if 'substitution' in C:         C['substitution'][rg][e:f]   += 1.0
                            if cigar_list[i][0]==3:
                                if 'splice' in C:               C['splice'][rg][e:f]         += 1.0
            #cigar ops based calculations---------------------
    samfile.close()
    #'total','primary','proper_pair','discordant','mapq','mapq_pp','mapq_dis','tlen','tlen_pp','tlen_dis','len_diff','tlen_rd','big_del','RD','GC'
    if ('left_smap_same' in C or 'left_smap_diff' in C or\
            'right_smap_same' in C or 'right_smap_diff' in C or\
            'smap_same' in C or 'smap_diff' in C) and ref_seq is not None: #minimap2=>mappy based soft-clipped re-alignment
        al = mappy.Aligner(seq=ref_seq,preset=align_preset,n_threads=0)
        print('aligning %s split read fragments to seq=%s from rgs=%s'%(sum([len(S[rg]) for rg in S]),seq,','.join(list(S.keys()))))
        for rg in S:
            A = {}
            for k in S[rg]:
                A[k] = []
                for i in range(len(S[rg][k])):
                    for hit in al.map(S[rg][k][i][0]):
                        if hit.mapq>min_smapq: #decent mapq please placement probability of 0.99
                            o_start  = S[rg][k][i][2]
                            o_strand = S[rg][k][i][4]
                            n_mapq   = hit.mapq   #certainty of the pos
                            n_start  = hit.r_st   #reference start position
                            n_end    = hit.r_en   #reference end position
                            n_mlen   = hit.mlen   #length of the match
                            n_strand = hit.strand #forward 1 or reverse -1
                            if n_strand<0: n_start,n_end = n_end,n_start
                            c_l      = n_start-o_start+1
                            if c_l<0: o_start,n_start = n_start,o_start
                            if abs(c_l)>n_mlen: #smap distance has to be further than the match length
                                A[k] += [[o_start,n_start,c_l,n_strand==o_strand]]
            print('%s split read fragments were aligned to seq=%s with mapq>%s'%(len(A),seq,min_smapq))
            for k in A:
                for i in range(len(A[k])):
                    e = max(0,min(A[k][i][0]-start,end))
                    f = max(0,min(A[k][i][1]-start,end))
                    if e>f: f,e = e,f
                    if A[k][i][3]:
                        if k=='L':
                            if 'left_smap_same' in C:  C['left_smap_same'][rg][e:f]  += 1.0
                        else:
                            if 'right_smap_same' in C: C['right_smap_same'][rg][e:f] += 1.0
                        if 'smap_same' in C:           C['smap_same'][rg][e:f] += 1.0
                    else:
                        if k=='L':
                            if 'left_smap_diff' in C:  C['left_smap_diff'][rg][e:f]  += 1.0
                        else:
                            if 'right_smap_diff' in C: C['right_smap_diff'][rg][e:f] += 1.0
                        if 'smap_diff' in C:           C['smap_diff'][rg][e:f] += 1.0
    if not merge_rg:
        for rg in sms:
            if 'RD' in C and 'primary' in C and 'GC' in C and 'total' in C:
                C['RD'][rg]              = (2.0*C['primary'][rg])*(C['GC'][rg]/C['total'][rg])
                if 'mapq' in C:
                    C['MD'][rg]          = C['RD'][rg]*(np.clip((C['mapq'][rg]/C['total'][rg]-1.0)/60.0,0.0,1.0))
            if 'GC' in C:
                C['GC'][rg]              = C['GC'][rg]-1.0
            if 'mapq' in C:
                C['mapq'][rg]            = C['mapq'][rg]/C['total'][rg]-1.0
            if 'mapq_pp' in C:
                C['mapq_pp'][rg]         = C['mapq_pp'][rg]/C['proper_pair'][rg]-1.0
            if 'mapq_dis' in C:
                C['mapq_dis'][rg]        = C['mapq_dis'][rg]/C['discordant'][rg]-1.0
            if 'tlen' in C:
                C['tlen'][rg]            = C['tlen'][rg]/C['total'][rg]-1.0
            if 'orient_same_rd' in C:
                C['orient_same_rd'][rg]  = C['orient_same_rd'][rg]/C['total'][rg]-1.0
            if 'orient_out_rd' in C:
                C['orient_out_rd'][rg]   = C['orient_out_rd'][rg]/C['total'][rg]-1.0
            if 'tlen_pp' in C:
                C['tlen_pp'][rg]         = C['tlen_pp'][rg]/C['proper_pair'][rg]-1.0
            if 'tlen_pp_rd' in C:
                C['tlen_pp_rd'][rg]      = C['tlen_pp_rd'][rg]/C['proper_pair'][rg]-1.0
            if 'tlen_dis' in C:
                C['tlen_dis'][rg]        = C['tlen_dis'][rg]/C['discordant'][rg]-1.0
            if 'tlen_dis_rd' in C:
                C['tlen_dis_rd'][rg]     = C['tlen_dis_rd'][rg]/C['discordant'][rg]-1.0
            if 'discordant' in C:
                C['discordant'][rg]      = C['discordant'][rg]-1.0
            if 'proper_pair' in C:
                C['proper_pair'][rg]     = C['proper_pair'][rg]-1.0
            if 'primary' in C:
                C['primary'][rg]         = C['primary'][rg]-1.0
            if 'total' in C:
                C['total'][rg]           = C['total'][rg]-1.0
            if 'smap_same' in C:
                C['smap_same'][rg]       = C['smap_same'][rg]-1.0
            if 'smap_same' in C:
                C['smap_diff'][rg]       = C['smap_diff'][rg]-1.0
            if 'left_smap_same' in C:
                C['left_smap_same'][rg]  = C['left_smap_same'][rg]-1.0
            if 'left_smap_diff' in C:
                C['left_smap_diff'][rg]  = C['left_smap_diff'][rg]-1.0
            if 'right_smap_same' in C:
                C['right_smap_same'][rg] = C['right_smap_same'][rg]-1.0
            if 'right_smap_diff' in C:
                C['right_smap_diff'][rg] = C['right_smap_diff'][rg]-1.0
    else:
        if 'RD' in C and 'primary' in C and 'GC' in C and 'total' in C:
                C['RD'][rg]              = (2.0*C['primary'][rg])*(C['GC'][rg]/C['total'][rg])
                if 'mapq' in C:
                    C['MD'][rg]          = C['RD'][rg]*(np.clip((C['mapq'][rg]/C['total'][rg]-1.0)/60.0,0.0,1.0))
        if 'GC' in C:
            C['GC'][rg]              = C['GC'][rg]-1.0
        if 'mapq' in C:
            C['mapq'][rg]            = C['mapq'][rg]/C['total'][rg]-1.0
        if 'mapq_pp' in C:
            C['mapq_pp'][rg]         = C['mapq_pp'][rg]/C['proper_pair'][rg]-1.0
        if 'mapq_dis' in C:
            C['mapq_dis'][rg]        = C['mapq_dis'][rg]/C['discordant'][rg]-1.0
        if 'tlen' in C:
            C['tlen'][rg]            = C['tlen'][rg]/C['total'][rg]-1.0
        if 'orient_same_rd' in C:
            C['orient_same_rd'][rg]  = C['orient_same_rd'][rg]/C['total'][rg]-1.0
        if 'orient_out_rd' in C:
            C['orient_out_rd'][rg]   = C['orient_out_rd'][rg]/C['total'][rg]-1.0
        if 'tlen_pp' in C:
            C['tlen_pp'][rg]         = C['tlen_pp'][rg]/C['proper_pair'][rg]-1.0
        if 'tlen_pp_rd' in C:
            C['tlen_pp_rd'][rg]      = C['tlen_pp_rd'][rg]/C['proper_pair'][rg]-1.0
        if 'tlen_dis' in C:
            C['tlen_dis'][rg]        = C['tlen_dis'][rg]/C['discordant'][rg]-1.0
        if 'tlen_dis_rd' in C:
            C['tlen_dis_rd'][rg]     = C['tlen_dis_rd'][rg]/C['discordant'][rg]-1.0
        if 'discordant' in C:
            C['discordant'][rg]      = C['discordant'][rg]-1.0
        if 'proper_pair' in C:
            C['proper_pair'][rg]     = C['proper_pair'][rg]-1.0
        if 'primary' in C:
            C['primary'][rg]         = C['primary'][rg]-1.0
        if 'total' in C:
            C['total'][rg]           = C['total'][rg]-1.0
        if 'smap_same' in C:
            C['smap_same'][rg]       = C['smap_same'][rg]-1.0
        if 'smap_same' in C:
            C['smap_diff'][rg]       = C['smap_diff'][rg]-1.0
        if 'left_smap_same' in C:
            C['left_smap_same'][rg]  = C['left_smap_same'][rg]-1.0
        if 'left_smap_diff' in C:
            C['left_smap_diff'][rg]  = C['left_smap_diff'][rg]-1.0
        if 'right_smap_same' in C:
            C['right_smap_same'][rg] = C['right_smap_same'][rg]-1.0
        if 'right_smap_diff' in C:
            C['right_smap_diff'][rg] = C['right_smap_diff'][rg]-1.0
    return C

#[0] utilities
@cython.boundscheck(False)
@cython.nonecheck(False)
def LRF_1D(list C1, list C2):
    cdef long n1,n2,j1,j2,upper
    cdef list I,U,D1,D2
    j1,j2,upper = 0,0,0         #initializations and padding
    n1,n2 = len(C1)+1,len(C2)+1 #boundries here
    I,U,  = [[-2,-2]],[[-2,-2]]
    D1,D2 = [[-2,-2]],[[-2,-2]]
    if n1 > 1 and n2 > 1:
        upper = max(C1[-1][1],C2[-1][1])
        C1 += [[upper+2,upper+2],[upper+4,upper+4]] #pad out the end of C1
        C2 += [[upper+2,upper+2],[upper+4,upper+4]] #pad out the end of C2
        while j1+j2 < n1+n2:  #pivioting dual ordinal indecies scan left to right on C1, C2
            a = C1[j1][0]-C2[j2][0]
            b = C1[j1][0]-C2[j2][1]
            c = C1[j1][1]-C2[j2][0]
            d = C1[j1][1]-C2[j2][1]
            if    C1[j1][0:2]==C2[j2][0:2]: #[7] c1 and c2 are equal on x
                orientation_7(C1,j1,C2,j2,I,U,D1,D2)
                j1 += 1
                j2 += 1
            elif  c<0:                      #[1] c1 disjoint of left of c2
                orientation_1(C1,j1,C2,j2,I,U,D1,D2)
                j1 += 1
            elif  b>0:                      #[6] c1 disjoint right of c2
                orientation_6(C1,j1,C2,j2,I,U,D1,D2)
                j2 += 1
            elif  a<0 and d<0:              #[2] c1 right overlap to c2 left no envelopment
                orientation_2(C1,j1,C2,j2,I,U,D1,D2)
                j1 += 1
            elif  a>0 and d>0:              #[4] c1 left overlap of c2 right no envelopment
                orientation_4(C1,j1,C2,j2,I,U,D1,D2)
                j2 += 1
            elif  a<=0 and d>=0:            #[3] c1 envelopment of c2
                orientation_3(C1,j1,C2,j2,I,U,D1,D2)
                j2 += 1
            elif  a>=0 and d<=0:            #[5] c1 enveloped by c2
                orientation_5(C1,j1,C2,j2,I,U,D1,D2)
                j1 += 1
            if j1>=n1: j1,j2 = n1,j2+1 #sticky indecies wait for eachother
            if j2>=n2: j2,j1 = n2,j1+1 #sticky indecies wait for eachother
        #pop off extras for each features (at most two at the end)
        while len(C1) > 0 and C1[-1][0] > upper:  C1.pop()
        while len(C2) > 0 and C2[-1][0] > upper:  C2.pop()
        while len(I)  > 0 and I[-1][0]>upper:      I.pop()
        if len(I) > 0 and I[-1][1]>upper:         I[-1][1] = upper
        while len(U) > 0 and U[-1][0]>upper:      U.pop()
        if len(U) > 0 and U[-1][1]>upper:         U[-1][1] = upper
        while len(D1) > 0 and D1[-1][0]>upper:    D1.pop()
        if len(D1) > 0 and D1[-1][1]>upper:       D1[-1][1] = min(C2[-1][0]-1,C1[-1][1])
        while len(D2) > 0 and D2[-1][0]>upper:    D2.pop()
        if len(D2) > 0 and D2[-1][1]>upper:       D2[-1][1] = min(C1[-1][0]-1,C2[-1][1])
    else:
        if   n1==1:
            if n2>1: U,D2 = U+C2,D2+C2
        elif n2==1:
            if n1>1: U,D1 = U+C1,D1+C1
    return I[1:],U[1:],D1[1:],D2[1:]

#[1] c1 disjoint of left of c2
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void orientation_1(list C1,long j1,list C2,long j2,list I,list U,list D1,list D2):
    #[i]----------------------[i] #no intersection
    #[u]----------------------[u]
    if U[-1][1]+1 >= C1[j1][0]:
        U[-1][1] = C1[j1][1]
    else:
        U += [[C1[j1][0],C1[j1][1]]]
    #[d1]--------------------[d1]
    if D1[-1][1]+1 >= C1[j1][0]:  #extend segment
        if D1[-1][1]+1!=C2[j2-1][0]:
            D1[-1][1] = C1[j1][1]
    else:                         #new segment
        D1 += [[C1[j1][0],C1[j1][1]]]
    #[d2]--------------------[d2] #no set two difference

#[2] c1 right overlap to c2 left no envelopment
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void orientation_2(list C1,long j1,list C2,long j2,list I,list U,list D1,list D2):
    #[i]----------------------[i]
    if I[-1][1]+1 >= C2[j2][0]:
        I[-1][1] = C1[j1][1] #was C2[j2][1]
    else:
        I += [[C2[j2][0],C1[j1][1]]]
    #[u]----------------------[u]
    if U[-1][1]+1 >= C1[j1][0]:
        U[-1][1] = C2[j2][1]
    else:
        U += [[C1[j1][0],C2[j2][1]]]
    #[d1]--------------------[d1]
    if D1[-1][1]+1 >= C1[j1][0]:
        D1[-1][1] = C1[j1][1]
        if D1[-1][1] > C2[j2][0]-1:
            D1[-1][1] = C2[j2][0]-1
            if D1[-1][1] < D1[-1][0]: D1.pop()
    else:
        D1 += [[C1[j1][0],C2[j2][0]-1]]
    #[d2]--------------------[d2]
    D2 += [[C1[j1][1]+1,C2[j2][1]]]

#[3] c1 envelopment of c2
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void orientation_3(list C1,long j1,list C2,long j2,list I,list U,list D1,list D2):
    #[i]----------------------[i]
    if I[-1][1]+1 >= C2[j2][0]:
        I[-1][1] = C2[j2][1]
    else:
        I += [[C2[j2][0],C2[j2][1]]]
    #[u]----------------------[u]
    if U[-1][1]+1 >= C1[j1][0]:
        U[-1][1] = C1[j1][1]
    else:
        U += [[C1[j1][0],C1[j1][1]]]
    #[d1]--------------------[d1]
    if D1[-1][1]+1 >= C1[j1][0]:
        D1[-1][1] = C2[j2][0]-1
        if C2[j2][1] < C1[j1][1]:
            D1 += [[C2[j2][1]+1,C1[j1][1]]]
    elif D1[-1][1] >= C2[j2][0]:
        D1[-1][1] = C2[j2][0]-1
        if C2[j2][1] < C1[j1][1]:  #has a right side
            D1 += [[C2[j2][1]+1,C1[j1][1]]]
    else:
        if C1[j1][0] < C2[j2][0]:  #has a left side
            D1 += [[C1[j1][0],C2[j2][0]-1]]
        if C2[j2][1] < C1[j1][1]:  #has a right side
            D1 += [[C2[j2][1]+1,C1[j1][1]]]


#[4] c1 left overlap of c2 right no envelopment
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void orientation_4(list C1,long j1,list C2,long j2,list I,list U,list D1,list D2):
    #[i]----------------------[i]
    if I[-1][1]+1 >= C1[j1][0]:
        I[-1][1] = C2[j2][1]
    else:
        I += [[C1[j1][0],C2[j2][1]]]
    #[u]----------------------[u]
    if U[-1][1]+1 >= C2[j2][0]:
        U[-1][1] = C1[j1][1]
    else:
        U += [[C2[j2][0],C1[j1][1]]]
    #[d1]--------------------[d1]
    D1 += [[C2[j2][1]+1,C1[j1][1]]]
    #[d2]--------------------[d2]
    if D2[-1][1]+1 >= C2[j2][0]:
        D2[-1][1] = C2[j2][1]
        if D2[-1][1] > C1[j1][0]-1:
            D2[-1][1] = C1[j1][0]-1
            if D2[-1][1] < D2[-1][0]: D2.pop()
    else:
        D2 += [[C2[j2][0],C1[j1][0]-1]]


#[5] c1 enveloped by c2
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void orientation_5(list C1,long j1,list C2,long j2,list I,list U,list D1,list D2):
    #[i]----------------------[i]
    if I[-1][1]+1 >= C1[j1][0]:
        I[-1][1] = C1[j1][1]
    else:
        I += [[C1[j1][0],C1[j1][1]]]

    #[u]----------------------[u]
    if U[-1][1]+1 >= C2[j2][0]:
        U[-1][1] = C2[j2][1]
    else:
        U += [[C2[j2][0],C2[j2][1]]]
    #[d1]--------------------[d1] #no set one difference
    #[d2]--------------------[d2]
    if D2[-1][1]+1 >= C2[j2][0]:
        D2[-1][1] = C1[j1][0]-1
        if C1[j1][1] < C2[j2][1]:
            D2 += [[C1[j1][1]+1,C2[j2][1]]]
    elif D2[-1][1] >= C1[j1][0]:
        D2[-1][1] = C1[j1][0]-1
        if C1[j1][1] < C2[j2][1]:  #has a right side
            D2 += [[C1[j1][1]+1,C2[j2][1]]]
    else:
        if C2[j2][0] < C1[j1][0]:  #has a left side
            D2 += [[C2[j2][0],C1[j1][0]-1]]
        if C1[j1][1] < C2[j2][1]:  #has a right side
            D2 += [[C1[j1][1]+1,C2[j2][1]]]

#[6] c1 disjoint right of c2
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void orientation_6(list C1,long j1,list C2,long j2,list I,list U,list D1,list D2):
    #[i]----------------------[i] #no instersection
    if U[-1][1]+1 >= C2[j2][0]:
        U[-1][1] = C2[j2][1]
    else:
        U += [[C2[j2][0],C2[j2][1]]]
    #[d1]--------------------[d1] #no set one difference
    #[d2]--------------------[d2]
    if D2[-1][1]+1 >= C2[j2][0]:
        if D2[-1][1]+1!=C1[j1-1][0]:
            D2[-1][1] = C2[j2][1]
    else:
        D2 += [[C2[j2][0],C2[j2][1]]]

#[7] c1 and c2 are equal on x
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void orientation_7(list C1,long j1,list C2,long j2,list I,list U,list D1,list D2):
    #[i]----------------------[i]
    if I[-1][1]+1 >= C1[j1][0]:
        I[-1][1] = C1[j1][1]
    else:
        I += [[C1[j1][0],C1[j1][1]]]

    #[u]----------------------[u]
    if U[-1][1]+1 >= C1[j1][0]:
        U[-1][1] = C1[j1][1]
    else:
        U += [[C1[j1][0],C1[j1][1]]]
    #[d1]----------------------[d1]
    #[d2]----------------------[d2]

#[I]------------statistical features--------------------[I]#
#notes: median and entropy are removed for now
#median can be estimated from the hist (if linear...)
#entropy is captured to some degree by transistions...

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def standardized_moments(double[:,:] X):
    cdef unsigned int n,i,j
    n = len(X)
    cdef np.ndarray[double, ndim=2] D = np.zeros([n,4],dtype=np.float64)
    for i in range(n):
        D[i][0] = X[i][M1]                                                                         #sample mean
        D[i][1] = (math.pow(X[i][M2]/(X[i][N]-1),0.5) if X[i][N]>1 else 0.0)                       #sample std
        D[i][2] = (math.pow(X[i][N],0.5)*X[i][M3]/math.pow(X[i][M2],1.5) if X[i][M2]>0.0 else 0.0) #sample skew
        D[i][3] = (X[i][N]*X[i][M4]/(X[i][M2]*X[i][M2])-3.0 if X[i][M2]>0.0 else -3.0)             #sample kur
    return D

#F[0=n,1=sum,2=min,3=max,4=M1,5=M2,6=M3,7=M4]
#standard 2-pass exact algorithm-----------------------------------------
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def exact_moments(float[::1] X, unsigned int x_i, unsigned int x_j,
                  double[::1] Y, unsigned int y_i):
    cdef unsigned int i         #main loop index
    cdef float mx = np.finfo(np.float32).min  #smallest for float
    cdef float mn = np.finfo(np.float32).max  #largest for float
    cdef double m               #total window range
    cdef double rm[4]           #difference
    cdef double srm[4]          #accumulators
    srm[:] = [0.0,0.0,0.0,0.0]
    m = <double>x_j-<double>x_i
    Y[y_i+N] = m
    #starting sum/min/max/mean
    for i in range(x_i,x_j):
        srm[0] += X[i]
        if X[i] > mx: mx = X[i]
        if X[i] < mn: mn = X[i]
    #do the diff powers all together and normalize once
    if srm[0] > 0.0:
        Y[y_i+SUM] = srm[0]
        Y[y_i+MIN] = <double>mn
        Y[y_i+MAX] = <double>mx
        Y[y_i+M1] = srm[0]/m
        for i in range(x_i,x_j):
            rm[0]   = X[i]-Y[y_i+M1] #(xi-xbar)**1
            rm[1]   = rm[0]*rm[0]    #(xi-xbar)**2
            rm[2]   = rm[0]*rm[1]    #(xi-xbar)**3
            rm[3]   = rm[0]*rm[2]    #(xi-xbar)**4
            srm[1] += rm[1]          #j=0:i-1 + (xi-xbar)**2
            srm[2] += rm[2]          #j=0:i-1 + (xi-xbar)**3
            srm[3] += rm[3]          #j=0:i-1 + (xi-xbar)**4
        Y[y_i+M2] = srm[1]
        if Y[y_i+M2] > 0.0:
            Y[y_i+M3] = srm[2]
            Y[y_i+M4] = srm[3]
        else:
            Y[y_i+M4] = -3.0
    else:
        Y[y_i+M4] = -3.0
#[I]------------statistical features--------------------[I]#

#[A]------------sliding statistical features------------[A]#
#notes: median and entropy are removed for now
#designed for similiar accuracy to features above but
#with data at every position inside a buffer inside a window

#does running low mememory calculation of a window
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def running_moments(float[::1] X, unsigned int x_i, unsigned int x_j,
                    double[::1] Y, unsigned int y_i):
    cdef unsigned int n
    cdef double v,x,d,d_n,dd_nn,d_n2
    Y[y_i+N],Y[y_i+MIN],Y[y_i+MAX] = <double>x_j-x_i,<double>X[x_i],<double>X[x_i]
    for n in range(x_j-x_i):
        x,v = <double>X[x_i+n],<double>(n+1)
        Y[y_i+SUM] += x
        d           = x - Y[y_i+M1]
        d_n         = d/v
        dd_nn       = d*d_n*(v-1)
        d_n2        = d_n*d_n
        Y[y_i+M4]  += dd_nn*d_n2*(v*v-3*v+3) + 6*d_n2*Y[y_i+M2] - 4*d_n*Y[y_i+M3]
        Y[y_i+M3]  += dd_nn*d_n*(v-2) - 3*d_n*Y[y_i+M2]
        Y[y_i+M2]  += dd_nn
        Y[y_i+M1]  += d_n
        if x < Y[y_i+MIN]: Y[y_i+MIN] = x
        if x > Y[y_i+MAX]: Y[y_i+MAX] = x
        #R/W without a GIL and shared mem

# update: + (new point - old moment) - (old point - new moment)
#now sliding algorithm---------------------------------------------------------------------
#X is the input array, D is a dynamic programming table for min/max, Y is the result array
#x_i, x_j is the view indecies or start stop, w is the sliding window size, r is the result value
#x_i is used for the dynamic programming table D that is 4*len(X) and for the features results FN*x_i+N...
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def sliding_moments(float[::1] X, unsigned int x_i, unsigned int x_j,
                    unsigned int w, unsigned int r, double[::1] Y):
    cdef unsigned int i,j,x
    cdef double r_d,r_dn,r_dd_nn,r_dn2,l_d,l_dn,l_dd_nn,l_dn2,v,v_p
    cdef long double r_dd,l_dd
    cdef np.ndarray[float, ndim=1] D = np.zeros([4*(<long>x_j-<long>x_i),], dtype=np.float32)
    v = <double>w       #cast v to save conversion operations
    v_p = v*v-<double>3.0*v+<double>3.0 #upper moments terms is cached
    r   = r+x_i+1       #hard reset
    if x_j>x_i and (x_j-x_i)>w: #need at least one value here which is greater than the window size
        x = <unsigned int>(<long>x_j-<long>x_i)
        D[4*x_i+0]     = D[4*x_i+1]     = X[x_i+0]
        D[4*(x_j-1)+2] = D[4*(x_j-1)+3] = X[x_j-1]
        for i in range(x_i+1,x_j):
            j = (x-i-1)
            if i%w==0: D[4*i+0],D[4*i+1] = X[i],X[i]
            else:      D[4*i+0],D[4*i+1] = min(D[4*(i-1)+0],X[i]),max(D[4*(i-1)+1],X[i])
            if j%w==0: D[4*j+2],D[4*j+3] = X[j],X[j]
            else:      D[4*j+2],D[4*j+3] = min(D[4*(j+1)+2],X[j]),max(D[4*(j+1)+3],X[j])
        exact_moments(X,x_i,x_i+w,Y,FN*x_i) #first------------------------------------------------------------------
        for i in range(x_i+1,x_j-w):
            Y[FN*i+MIN]       = min(D[4*(i+w-1)+0],D[4*i+2])
            Y[FN*i+MAX]       = max(D[4*(i+w-1)+1],D[4*i+3])
        for i in range(x_i+1,x_j-w): #no conditonals in the main loop
            Y[FN*i+N]       = v
            Y[FN*i+SUM]     = Y[FN*(i-1)+SUM] + <double>X[i+w-1] - <double>X[i-1]
            Y[FN*i+M1]      = Y[FN*i+SUM]/v
            r_d,l_d         = <double>X[i+w-1]-Y[FN*(i-1)+M1], <double>X[i-1]-Y[FN*i+M1]
            r_dn,l_dn       = r_d/v,l_d/v
            r_dd_nn,l_dd_nn = r_d*r_dn*(v+<double>1.0),l_d*l_dn*(v+<double>1.0)
            r_dd,l_dd       = <long double>r_d*<long double>r_d,<long double>l_d*<long double>l_d
            r_dn2,l_dn2     = r_dn**2,l_dn**2
            Y[FN*i+M2]      = max(<double>0.0,Y[FN*(i-1)+M2] + <double>(r_dd - l_dd))
            Y[FN*i+M3]      = Y[FN*(i-1)+M3] - <double>3.0*r_dn*Y[FN*(i-1)+M2] + <double>3.0*l_dn*Y[FN*i+M2] \
                              + r_dd_nn*r_dn*(v-<double>2.0) - l_dd_nn*l_dn*(v-<double>2.0)
            Y[FN*i +M4]     = Y[FN*(i-1) +M4] - 4*(r_dn)*Y[FN*(i-1) +M3] + 4.0*(l_dn)*Y[FN*i +M3] \
                              + <double>6.0*r_dn2*Y[FN*(i-1) +M2] - <double>6.0*l_dn**2*Y[FN*i +M2] \
                              + r_dd_nn*r_dn2*v_p - l_dd_nn*l_dn**2*v_p
            if i%r==0: exact_moments(X,i,i+w,Y,FN*i)
        exact_moments(X,x_j-2*w,x_j-w,Y,FN*(x_j-2*w))
        #R/W without a GIL and shared mem
#[A]------------sliding statistical features------------[A]#

#merges all the disjoint window moments from pre-existing sliding Y into Z with windows 2*w
#assumes that Y is a single window moment structure
#add terminal condition that checks when a_n is the full boundry...
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_sliding_moments_binary(double[::1] Y, double[::1] Z):
    cdef unsigned int a,b,i,y,w
    cdef double d1,d2,d3,d4,a_n,a_m1,a_m2,a_m3,a_m4,b_n,b_m1,b_m2,b_m3,b_m4
    if len(Y)>0:
        y,w = <unsigned int>(len(Y)/FN),<unsigned int>Y[0]
        if 2*w <= y:
            for i in range(y-w):
                a,b = FN*i,FN*(i+w)
                a_n,a_m1,a_m2,a_m3,a_m4 = Y[a+N],Y[a+M1],Y[a+M2],Y[a+M3],Y[a+M4]
                b_n,b_m1,b_m2,b_m3,b_m4 = Y[b+N],Y[b+M1],Y[b+M2],Y[b+M3],Y[b+M4]
                Z[FN*i+N]   = a_n+b_n
                Z[FN*i+SUM] = Y[a+SUM] + Y[b+SUM]
                Z[FN*i+MIN] = min(Y[a+MIN],Y[b+MIN])
                Z[FN*i+MAX] = max(Y[a+MAX],Y[b+MAX])
                d1          = b_m1 - a_m1
                d2          = d1*d1
                d3          = d1*d2
                d4          = d2*d2
                Z[FN*i+M1]  = (a_n*a_m1 + b_n*b_m1)/Z[FN*i+N]
                Z[FN*i+M2]  = a_m2 + b_m2 + d2*a_n*b_n/Z[FN*i+N]
                Z[FN*i+M3]  = a_m3 + b_m3 + d3*a_n*b_n*(a_n-b_n)/(Z[FN*i+N]**2) \
                              + <double>3.0*d1*(a_n*b_m2 - b_n*a_m2)/Z[FN*i+N]
                Z[FN*i+M4]  = a_m4 + b_m4 \
                              + d4*a_n*b_n*(a_n**2 - a_n*b_n + b_n**2)/(Z[FN*i+N]**3) \
                              + <double>6.0*d2*((a_n**2)*b_m2 + (b_n**2)*a_m2)/(Z[FN*i+N]**2) \
                              + <double>4.0*d1*(a_n*b_m3 - b_n*a_m3)/Z[FN*i+N]
    #R/W without a GIL and in shared mem

#merges disjoint window moments from preexisting sliding Y into Z with a target window size
#will operate and cascade writes back into Y if several merges need to be completed (AKA Bennet)
#provide that base window sizes are in Y[0]=>w  t is rounded to the next highest 
#multiple of w that allows features to be constructed +. t>w
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_sliding_moments_target(double[::1] Y, double[::1] Z, unsigned int t):
    cdef unsigned int a,b,i,j,y,w,v
    cdef double d1,d2,d3,d4,a_n,a_m1,a_m2,a_m3,a_m4,b_n,b_m1,b_m2,b_m3,b_m4
    cdef np.ndarray[double, ndim=1] T = np.zeros([FN,], dtype=np.float64)
    if len(Y)>0 and <unsigned int>Y[0]<t and <unsigned int>Y[0]>0:
        y,w = <unsigned int>(len(Y)/FN),<unsigned int>Y[0] #length of features and window
        v = <unsigned int>(t/w)                            #closest integer of widows needed
        if t%w>1: v += 1                     #to construct the target
        if v*w <= y:                         #leave out boundries that are too small
            for i in range(y-(v-1)*w):       #total number of windows in Y
                for j in range(FN): T[j] = Y[FN*i+j] #write the first partition to T
                for j in range(1,v):         #now accumulate the tiles into T
                    b = FN*(i+j*w)           #b is the growing tiles
                    a_n,a_m1,a_m2,a_m3,a_m4 = T[N],T[M1],T[M2],T[M3],T[M4]
                    b_n,b_m1,b_m2,b_m3,b_m4 = Y[b+N],Y[b+M1],Y[b+M2],Y[b+M3],Y[b+M4]
                    T[N]   = a_n+b_n
                    T[SUM] = T[SUM] + Y[b+SUM]
                    T[MIN] = min(T[MIN],Y[b+MIN])
                    T[MAX] = max(T[MAX],Y[b+MAX])
                    d1     = b_m1 - a_m1
                    d2     = d1*d1
                    d3     = d1*d2
                    d4     = d2*d2
                    T[M1]  = (a_n*a_m1 + b_n*b_m1)/T[N]
                    T[M2]  = a_m2 + b_m2 + d2*a_n*b_n/T[N]
                    T[M3]  = a_m3 + b_m3 + d3*a_n*b_n*(a_n-b_n)/(T[N]**2) \
                             + <double>3.0*d1*(a_n*b_m2 - b_n*a_m2)/T[N]
                    T[M4]  = a_m4 + b_m4 \
                             + d4*a_n*b_n*(a_n**2 - a_n*b_n + b_n**2)/(T[N]**3) \
                             + <double>6.0*d2*((a_n**2)*b_m2 + (b_n**2)*a_m2)/(T[N]**2) \
                             + <double>4.0*d1*(a_n*b_m3 - b_n*a_m3)/T[N]
                for j in range(FN): Z[FN*i+j] = T[j] #write out the accumulated results
            #R/W without a GIL and in shared mem           

#merges disjoint window moments from pre-existing tiled Y into tiled Z with a target window size t
#will operate and cascade writes back into Y if serveral merges need to be completed (AKA 2008 Los Almos Labs Pebay, Bennet)
#provide that base window sizes are Y[0]=>w t is rounded to the next highest
#multiple of w that allows features to be constructed +. t>w
#additionally a logrithmic data reduction and computational reduction step disjoint can be used
#if hierarchical summary only is desired and not sliding tile summary, having the effect of
#either merged target windows that overlap by (v-2) at each position or an overlap of 0.
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_tiled_moments_target(double[::1] Y, double[::1] Z, unsigned int t, bint disjoint=True):
    cdef unsigned int a,b,c,i,j,y,v,z
    cdef double d1,d2,d3,d4,a_n,a_m1,a_m2,a_m3,a_m4,b_n,b_m1,b_m2,b_m3,b_m4
    cdef np.ndarray[double, ndim=1] T = np.zeros([FN,], dtype=np.float64)
    if t>1 and Y[0]>0:
        y = <unsigned int>len(Y)//FN#len of features, window, targeted window multiple
        if disjoint: #---------------------------------------------------------------
            for i in range(0,y-t,t):
                for j in range(FN): T[j] = Y[FN*i+j] #grab first of the disjoint tiles
                for j in range(1,t):         #now accumulate the tiles into T
                    b = (i+j)*FN             #b is the growing tiles
                    a_n,a_m1,a_m2,a_m3,a_m4 = T[N],T[M1],T[M2],T[M3],T[M4]
                    b_n,b_m1,b_m2,b_m3,b_m4 = Y[b+N],Y[b+M1],Y[b+M2],Y[b+M3],Y[b+M4]
                    T[N]   = a_n+b_n
                    T[SUM] = T[SUM] + Y[b+SUM]
                    T[MIN] = min(T[MIN],Y[b+MIN])
                    T[MAX] = max(T[MAX],Y[b+MAX])
                    d1     = b_m1 - a_m1
                    d2     = d1*d1
                    d3     = d1*d2
                    d4     = d2*d2
                    T[M1]  = (a_n*a_m1 + b_n*b_m1)/T[N]
                    T[M2]  = a_m2 + b_m2 + d2*a_n*b_n/T[N]
                    T[M3]  = a_m3 + b_m3 + d3*a_n*b_n*(a_n-b_n)/(T[N]**2) \
                             + <double>3.0*d1*(a_n*b_m2 - b_n*a_m2)/T[N]
                    T[M4]  = a_m4 + b_m4 + d4*a_n*b_n*(a_n**2 - a_n*b_n + b_n**2)/(T[N]**3) \
                             + <double>6.0*d2*((a_n**2)*b_m2 + (b_n**2)*a_m2)/(T[N]**2) \
                             + <double>4.0*d1*(a_n*b_m3 - b_n*a_m3)/T[N]
                c = (i//t)*FN
                for j in range(FN): Z[c+j] = T[j]
            if y%t>0: #left over windows that are a remander of the target size
                i = y-(y%t)
                for j in range(FN): T[j] = Y[FN*i+j]
                for j in range(1,y%t):       #now accumulate the tiles into T
                    b = (i+j)*FN             #b is the growing tiles
                    a_n,a_m1,a_m2,a_m3,a_m4 = T[N],T[M1],T[M2],T[M3],T[M4]
                    b_n,b_m1,b_m2,b_m3,b_m4 = Y[b+N],Y[b+M1],Y[b+M2],Y[b+M3],Y[b+M4]
                    T[N]   = a_n+b_n
                    T[SUM] = T[SUM] + Y[b+SUM]
                    T[MIN] = min(T[MIN],Y[b+MIN])
                    T[MAX] = max(T[MAX],Y[b+MAX])
                    d1     = b_m1 - a_m1
                    d2     = d1*d1
                    d3     = d1*d2
                    d4     = d2*d2
                    T[M1]  = (a_n*a_m1 + b_n*b_m1)/T[N]
                    T[M2]  = a_m2 + b_m2 + d2*a_n*b_n/T[N]
                    T[M3]  = a_m3 + b_m3 + d3*a_n*b_n*(a_n-b_n)/(T[N]**2) \
                             + <double>3.0*d1*(a_n*b_m2 - b_n*a_m2)/T[N]
                    T[M4]  = a_m4 + b_m4 \
                             + d4*a_n*b_n*(a_n**2 - a_n*b_n + b_n**2)/(T[N]**3) \
                             + <double>6.0*d2*((a_n**2)*b_m2 + (b_n**2)*a_m2)/(T[N]**2) \
                             + <double>4.0*d1*(a_n*b_m3 - b_n*a_m3)/T[N]
                c = (i//t)*FN
                for j in range(FN): Z[c+j] = T[j]
        else:#---------------------------------------------------------------------
            for i in range(y-(t-1)):
                c = i*FN
                for j in range(FN): T[j] = Y[c+j]
                for j in range(1,t):         #now accumulate the tiles into T
                    b = (i+j)*FN             #b is the growing tiles
                    a_n,a_m1,a_m2,a_m3,a_m4 = T[N],T[M1],T[M2],T[M3],T[M4]
                    b_n,b_m1,b_m2,b_m3,b_m4 = Y[b+N],Y[b+M1],Y[b+M2],Y[b+M3],Y[b+M4]
                    T[N]   = a_n+b_n
                    T[SUM] = T[SUM] + Y[b+SUM]
                    T[MIN] = min(T[MIN],Y[b+MIN])
                    T[MAX] = max(T[MAX],Y[b+MAX])
                    d1     = b_m1 - a_m1
                    d2     = d1*d1
                    d3     = d1*d2
                    d4     = d2*d2
                    T[M1]  = (a_n*a_m1 + b_n*b_m1)/T[N]
                    T[M2]  = a_m2 + b_m2 + d2*a_n*b_n/T[N]
                    T[M3]  = a_m3 + b_m3 + d3*a_n*b_n*(a_n-b_n)/(T[N]**2) \
                             + <double>3.0*d1*(a_n*b_m2 - b_n*a_m2)/T[N]
                    T[M4]  = a_m4 + b_m4 \
                             + d4*a_n*b_n*(a_n**2 - a_n*b_n + b_n**2)/(T[N]**3) \
                             + <double>6.0*d2*((a_n**2)*b_m2 + (b_n**2)*a_m2)/(T[N]**2) \
                             + <double>4.0*d1*(a_n*b_m3 - b_n*a_m3)/T[N]
                for j in range(FN): Z[c+j] = T[j]
        #R/W without a GIL and in shared mem
            
#[II]#[I]------------spectrum features--------------------[II]#
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def spectrum_bin(float[::1] X, unsigned int x_i, unsigned int x_j,
                 float[::1] B, float[::1] Y, unsigned int y_len, unsigned int y_i):
    cdef unsigned int i,j
    cdef float w_x
    for i in range(x_i,x_j):
        w_x = X[i]
        for j in range(0,y_len):
            if w_x >= B[j] and w_x < B[j+1]:
                Y[j+y_i] += 1.0
                break             

#we assume Y is set to the correct size and the length of B is the number of bins desired
#we can assume where ever we are in X we are at the same position in Y via x_i
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def sliding_spectrum_bin(float[::1] X, unsigned int x_i,unsigned int x_j, unsigned int w,float[::1] B, float[::1] Y):
    cdef unsigned int i,a,k
    cdef float v_x,w_x
    k = len(B)-1 #one more bin boundry than the number of bins
    if x_j>x_i and (x_j-x_i)>w:
        spectrum_bin(X,x_i,x_i+w,B,Y,k,k*x_i) #first window---------------------------------------
        for i in range(x_i+1,x_j-w):
            v_x,w_x = X[i-1],X[i+w-1]
            for a in range(k):
                Y[i*k + a] = Y[(i-1)*k + a]
                if w_x >= B[a] and w_x < B[a+1]: Y[i*k + a] += 1.0  #update the incoming pos 
                if v_x >= B[a] and v_x < B[a+1]: Y[i*k + a] -= 1.0  #update the outgoing pos
    #R/W without a GIL and in shared mem        

#merges all disjoint window spectrum for pre-existing Y into Z with windows 2*w
#assumes that Y is a single window spectrum structure
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_spectrum_bin_binary(float[::1] Y, unsigned int w, float[::1] B, float[::1] Z):
    cdef unsigned int b,c,d,i,k,y
    b,y = len(B)-1,<unsigned int>(len(Y)/(len(B)-1))
    if y>0 and y > 2*w:
        for i in range(y-w):
            c,d = i*b,(i+w)*b
            for k in range(b):
                Z[i*k + c] = Y[i*k + c] + Y[i*k + d]
    #R/W without a GIL and in shared mem

#Y is the existing flat spectrum array input
#Z is the resulting merged flat spectrum array (can be the source due to buffering)?
#t is the target window size such that t>w +. v>0, ect
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_sliding_spectrum_bin_target(float[::1] Y, float[::1] Z, float[::1] B, unsigned int t):
    cdef unsigned b,c,d,i,j,k,y,v,w
    cdef np.ndarray[float, ndim=1] T = np.zeros([<unsigned int>(len(B)-1),], dtype=np.float32)
    w = <unsigned int>0
    for i in range(len(B)-1): w += <unsigned int>Y[i] 
    b,y,v = len(B)-1,len(Y)//(len(B)-1),<unsigned int>(t/w)
    if v>0 and y>0 and y>=v*w:
        for i in range(y-w*v):
            c = i*b
            for j in range(b): T[j] = Y[c+j]
            for j in range(1,v):
                d = c+j*w*b
                for k in range(b): T[k] += Y[d+k]
            for j in range(b): Z[c+j] = T[j]
    #R/W without a GIL and in shared mem

#Y is the existing flat spectrum array in tiled form, we assume that the windows are the same size
#Z is the output which can be disjoint for fully contained trees or as an exhaustive where
#every possible contigous set of t/w tiles are calculated and produced in the output
#B is the array of numerical boundries that were used to compute the spectrum
#t is the target window size such that t>w and disjoint sets the output behavior for Z    
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_tiled_spectrum_bin_target(float[::1] Y, float[::1] Z, float[::1] B, unsigned int t, bint disjoint=True):
    cdef unsigned b,c,d,i,j,k,y,v,w
    cdef np.ndarray[float, ndim=1] T = np.zeros([<unsigned int>(len(B)-1),], dtype=np.float32)
    w = <unsigned int>0
    for i in range(len(B)-1): w += <unsigned int>Y[i] 
    b,y,v = len(B)-1,<unsigned int>(len(Y)/(len(B)-1)),<unsigned int>(t/w)
    if t%w>1: v += 1
    if v>0 and y>0 and y>=v*w:
        if disjoint:#-------------------------------------
            for i in range(0,y-v,v): #the whole numebr of target windows
                c = i*b
                for j in range(b): T[j] = Y[c+j]
                for j in range(1,v):
                    d = (i+j)*b
                    for k in range(b): T[k] += Y[d+k]
                c = <unsigned int>(i/v)*b
                for j in range(b): Z[c+j] = T[j]
            if y%v>0: #left over windows that are less than t for one last window
                i = y-(y%v)
                for j in range(b): T[j] = Y[i*b+j]
                for j in range(1,y%v):      
                    d = (i+j)*b
                    for k in range(b): T[k] += Y[d+k]
                c = <unsigned int>(y/v)*b
                for j in range(b): Z[c+j] = T[j]  
        else:#---------------------------------------------------------------------
            for i in range(y-(v-1)): #every possibly merge in this version
                c = i*b
                for j in range(b): T[j] = Y[c+j]
                for j in range(1,v):         #now accumulate the tiles into T
                    d = (i+j)*b             #b is the growing tiles
                    for k in range(b): T[k] += Y[d+k]
                for j in range(b): Z[c+j] = T[j] 
    #R/W without a GIL and in shared mem
#[II]#[I]------------spectrum features--------------------[II]#

#[III]------------count transitions--------------------[III]#
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def transitions_bin(float[::1] X, unsigned int x_i, unsigned int x_j, float[::1] B,
                    float[::1] Y, unsigned int y_len, unsigned int y_i):
    cdef unsigned int i,j,k
    cdef float w_x,w_y
    for i in range(x_i,x_j):
        w_x = X[i]
        w_y = X[i+1]
        for j in range(0,y_len):
            for k in range(0,y_len):
                if w_x >= B[j] and w_x < B[j+1] and w_y >= B[k] and w_y < B[k+1]:
                    Y[y_len*k+j+y_i] += 1.0
                    break
            else:
                continue
            break
        
#we assume Y is set to the correct size and the length of B is the number of bins desired
#we can assume where ever we are in X we are at the same position in Y via x_i
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def sliding_transitions_bin(float[::1] X, unsigned int x_i,unsigned int x_j, unsigned int w,float[::1] B, float[::1] Y):
    cdef unsigned int a,b,i,k,kk
    cdef float v_x,v_y,w_x,w_y
    k = len(B)-1
    kk = k*k
    if x_j>x_i and (x_j-x_i)>w:
        transitions_bin(X,x_i,x_i+w-1,B,Y,k,k*x_i) #first window---------------------------------------
        for i in range(x_i+1,x_j-w-1): #check indecies here
            v_x,v_y,w_x,w_y = X[i-1],X[i],X[i+w-1],X[i+w]
            for a in range(k):                      
                for b in range(k):                
                    Y[i*kk + k*b + a] = Y[(i-1)*kk + k*b + a]                   
                    if w_x >= B[a] and w_x < B[a+1] and w_y >= B[b] and w_y < B[b+1]:
                        Y[i*kk + k*b + a] += 1.0   #update the incoming pos        
                    if v_x >= B[a] and v_x < B[a+1] and v_y >= B[b] and v_y < B[b+1]:
                        Y[i*kk + k*b + a] -= 1.0   #update the outgoing pos
    #R/W without a GIL and in shared mem

#merges all disjoint window spectrum for pre-existing Y into Z with windows 2*w
#assumes that Y is a single window spectrum structure
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_transitions_bin_binary(float[::1] Y, unsigned int w, float[::1] B, float[::1] Z):
    cdef unsigned int a,b,c,d,i,j,k,kk,y
    k = len(B)-1
    kk = k*k
    y = <unsigned int>(len(Y)/kk)
    if y>0 and y > 2*w:
        for i in range(y-w):
            for a in range(k):
                for b in range(k):
                    Z[i*kk + k*b + a] = Y[i*kk + k*b + a] + Y[(i+1)*kk + k*b + a]
    #R/W without a GIL and in shared mem

#take the sliding transitions data structure and merge all the disjoint windows up
#to the target window size t, buffering accumulations into T and
#storing into the output Z. this operation is buffered
#but since it is calculated left to right Y can be Z
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_sliding_transitions_bin_target(float[::1] Y, float[::1] Z, float[::1] B, unsigned int t):
    cdef unsigned int a,b,c,d,e,f,i,j,k,kk,y,w,v
    cdef np.ndarray[float, ndim=1] T = np.zeros([<unsigned int>(len(B)-1)**2,], dtype=np.float32)
    k = len(B)-1
    kk,w,y = k*k,<unsigned int>0,<unsigned int>(len(Y)/k*k)
    for i in range(kk): w += <unsigned int>Y[i] #window size in the first matrix
    v = <unsigned int>(t/w)                     #number of windows of size w that will make windows of size t
    if t%w>0: v += 1
    if v>0 and y>0 and y>=t*w:
        for i in range(y-v*w):
            c = i*kk                          #entry offset
            for j in range(kk): T[j] = Y[c+j] #copy first into T
            for j in range(1,v):
                d = c+j*w*kk
                for a in range(kk): T[a] += Y[d+a]
            for j in range(kk): Z[c+j] = T[j] #write result to Z (every 1-bp slide...)
    #R/W without a GIL and in shared mem
    
#:::TO DO:::---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def merge_tiled_transitions_bin_target(float[::1] Y, float[::1] Z, float[::1] B, unsigned int t, bint disjoint=True):
    cdef unsigned int a,b,c,d,e,f,i,j,k,kk,y,w,v
    cdef np.ndarray[float, ndim=1] T = np.zeros([<unsigned int>(len(B)-1)**2,], dtype=np.float32)
    k = len(B)-1
    kk,w,y = k*k,<unsigned int>0,<unsigned int>(len(Y)/k*k)
    for i in range(kk): w += <unsigned int>Y[i] #window size in the first matrix
    v = <unsigned int>(t/w)                     #number of windows of size w that will make windows of size t
#    if t%w>0: v += 1
#    if v>0 and y>0 and y>=t*w:
#        if disjoint:
#            for i in range(0,y-v,v):
#                c = i*kk
#                for j in range(kk): T[j] = Y[c+j]
#                for j in range(1,v):
#                    d = (i+j)*kk
#                    for a in range(kk): T[a] += Y[d+a]
#                c = <unsigned int>(i/v)*kk
#                for j in range(kk): Z[c+j] = T[j]
#
#            for i in range(0,y-v,v): #the whole numebr of target windows
#                c = i*b
#                for j in range(b): T[j] = Y[c+j]
#                for j in range(1,v):
#                    d = (i+j)*b
#                    for k in range(b): T[k] += Y[d+k]
#                c = <unsigned int>(i/v)*b
#                for j in range(b): Z[c+j] = T[j]
#            if y%v>0: #left over windows that are less than t for one last window
#                i = y-(y%v)
#                c = i*kk
#                for j in range(kk): T[j] = Y[c+j]
#                for j in
#                    for a in range(k):
#                        d = a*k
#                        for b in range(k): T[d+b] += Y[c+d+b]
#                
#                c = <unsigned int>(y/v)*kk
#                for j in range(kk): Z[c+j] = T[j]  
#        else:
#            
#    #R/W without a GIL and in shared mem
#    return True
#[III]------------count transitions--------------------[III]# 

#[IV]-----------normalization-and-pseudo-counting-----[IV]#
#[IV]-----------normalization-and-pseudo-counting------[IV]#

