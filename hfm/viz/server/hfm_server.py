import os
import glob
import json
import argparse
import math
from h5py import File
from aiohttp import web
import numpy as np

def read_json(path):
    data = {}
    with open(path,'r') as f:
        data = json.load(f)
    return data

def get_sample_meta(path):
    S = {}
    for hfm in glob.glob(path+'*.hdf5'):
        S[hfm] = {}
        f = File(hfm,'r')
        for sm in f:
            S[hfm][sm] = {}
            for rg in f[sm]:
                S[hfm][sm][rg] = {}
                for seq in f[sm][rg]:
                    S[hfm][sm][rg][seq] = {}
                    for trk in f[sm][rg][seq]:
                        S[hfm][sm][rg][seq][trk] = {}
                        for ftr in f[sm][rg][seq][trk]:
                            S[hfm][sm][rg][seq][trk][ftr] = sorted(list(f[sm][rg][seq][trk][ftr].keys()))
        f.close()
    return S

def query_meta(S,xs=None,sm=None,rg=None,seq=None,trk=None,ftr=None):
    Q = {}
    for x in S:
        if xs is None or x in xs:
            Q[x] = {}
            for s in S[x]:
                if sm is None or s in sm:
                    Q[x][s] = {}
                    for r in S[x][s]:
                        if rg is None or r in rg:
                            Q[x][s][r] = {}
                            for c in S[x][s][r]:
                                if seq is None or c in seq:
                                    Q[x][s][r][c] = {}
                                    for t in S[x][s][r][c]:
                                        if trk is None or t in trk:
                                            Q[x][s][r][c][t] = {}
                                            for f in S[x][s][r][c][t]:
                                                if ftr is None or f in ftr:
                                                    Q[x][s][r][c][t][f] = S[x][s][r][c][t][f]
    return Q

def merge_moments(Y):
    T = []
    N,SUM,MIN,MAX,M1,M2,M3,M4,FN=range(9)
    if(len(Y)>FN):
        y,d1,d2,d3,d4 = len(Y)/FN,0.0,0.0,0.0,0.0
        T = np.zeros((FN,),dtype=float)
        for j in range(FN): T[j] = Y[j]
        for i in range(1,y,1):
            b = i*FN
            a_n,a_m1,a_m2,a_m3,a_m4 = T[N],T[M1],T[M2],T[M4]
            b_n,b_m1,b_m2,b_m3,b_m4 = Y[b+N],Y[b+M1],Y[b+M2],Y[b+M3],Y[b+M4]
            T[N]   += Y[b+N]
            T[SUM] += Y[b+SUM];
            T[MIN] = min(T[MIN],Y[b+MIN])
            T[MAX] = max(T[MAX],Y[b+MAX])
            d1 = T[M1]-Y[b+M1]
            d2 = d1*d1
            d3 = d1*d2
            d4 = d2*d2
            T[M1] = (a_n*a_m1 + b_n*b_m1)/T[N]
            T[M2] = a_m2 + b_m2 + d2*a_n*b_n/T[N]
            T[M3] = a_m3 + b_m3 + d3*a_n*b_n*(a_n-b_n)/(T[N]**2)+\
                    3.0*d1*(a_n*b_m2-b_n*a_m2)/T[N]
            T[M4] = a_m4 + b_m4 + d4*a_n*b_n*(a_n**2 - a_n*b_n + b_n**2)/(T[N]**3)+\
                    6.0*d2*((a_n**2)*b_m2 + (b_n**2)*a_m2)/(T[N]**2)+\
                    4.0*d1*(a_n*b_m3 - b_n*a_m3)/T[N];
    return T

def buffer_seq_moments(seq,meta):
    return True

#meta is the query object, which means that sms shoul be selected out of it...
def buffer_moment_range(seq,start,end,meta,tiles=1600,FN=8,axis=0):
    range = abs(end-start)
    W = {}
    if axis<=0:
        for x in meta: #file
            file = File(x,'r')
            for s in meta[x]: #sm
                W[s] = {}
                for t in meta[x][s]['all'][seq]:  #seq is fixed in this IGV style API
                    W[s][t] = {}
                    best = [tiles,0]
                    for w in sorted(list(meta[x][s]['all'][seq][t]['moments'])):
                        window = int(w)
                        n = range/window
                        b = abs(tiles-n)
                        if n>-1 and n<=2*tiles and b<best[0]: best = [b,window]
                    y1 = math.floor(start/best[1])
                    y2 = math.ceil(end/best[1])
                    data = np.zeros((y2-y1,FN),dtype='f8')
                    data[:] = file[s]['all'][seq][t]['moments'][str(best[1])][y1:y2,:]
                    W[s][t] = data.tolist()
            file.close()
    else: #building axis information only
        i = 0
        try:
            x = list(meta.keys())[0]
            s = list(meta[x].keys())[0]
            t = list(meta[x][s]['all'][seq].keys())[0]
            best = [tiles,0]
            for w in sorted(list(meta[x][s]['all'][seq][t]['moments'])):
                window = int(w)
                n = range/window
                b = abs(tiles-n)
                if n>-1 and n<=2*tiles and b<best[0]: best = [b,window]
            y1 = math.floor(start/best[1])
            y2 = math.ceil(end/best[1])
            W['axis'] = {'length':y2-y1,'w':best[1]};
        except Exception as E:
            print('buff_moment_range: seq=%s start=%s end=%s tiles=%s\nmeta=%s\nerror hfm_server.py line 112:'%(seq,start,end,tiles,meta,E))
            pass
    return W

des = """HFM Vizualization Server, Copyright (C) 2020 Timothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--in_dir',type=str,help='hfm data input directory\t[None]')
args = parser.parse_args()
print(des)
if args.in_dir is not None: in_dir = args.in_dir
else: raise IOError

ref = read_json(in_dir+'/ref.meta.json')
if not os.path.exists(in_dir+'/sample.meta.json'):
    S = get_sample_meta(in_dir)
    with open(in_dir+'/sample.meta.json','w') as f:
        json.dump(S,f)
else:
    S = read_json(in_dir+'/sample.meta.json')
G = read_json(in_dir+'/gene.meta.json')
Genes = sorted(G.keys())
sample_files = {list(S[x].keys())[0]:x for x in S}

async def sample_map_h(request):
    return web.json_response(S)

async def gene_map_h(request):
    return web.json_response(G)

async def ref_map_h(request):
    return web.json_response(ref)

async def genes_h(request):
    return web.json_response(Genes)

#to do for full seq total trk only...
async def seq_h(request):
    response = {'seq':request.match_info.get('seq', "Anonymous")}
    return web.json_response(response)

async def gene_h(request):
    tiles,f_p,W = 1600,0.5,{}
    try:
        gene  = request.match_info.get('gene','Anonymous')
        seq,start,end = G[gene][0] #get only the first for now
        flank = max(0,int(round((end-start)*f_p)))
        W = buffer_moment_range(W['coord'][0],W['coord'][1],W['coord'][2],S,tiles=tiles,FN=8)
        W['coord'] = [seq,max(0,start-flank),min(end+flank,ref[seq])]
    except Exception as E:
        print('Gene Symbol %s was not found:\tline 167:%s'%(gene,E))
        W = {'err':'Gene Symbol %s was not found'%gene}
    return web.json_response(W)

async def gene_tiles_flank_h(request):
    W = {}
    try:
        gene  = request.match_info.get('gene','Anonymous')
        seq,start,end = G[gene][0] #get only the first for now
        tiles = int(request.match_info.get('tiles','Anonymous'))
        f_p   = float(request.match_info.get('flank','Anonymous'))
        axis  = int(request.match_info.get('axis','Anonymous'))
        flank = max(0,int(round((end-start)*f_p)))
        coord = seq,max(0,start-flank),min(end+flank,ref[seq])
        W = buffer_moment_range(coord[0],coord[1],coord[2],S,tiles=tiles,FN=8,axis=axis)
        W['coord'] = coord
    except Exception as E:
        print('Gene Symbol %s was not found:\tline 180:%s'%(gene,E))
        W = {'err':'Gene Symbol %s was not found'%gene}
    return web.json_response(W)

async def sm_gene_tiles_flank_h(request):
    W = {}
    try:
        sm    = request.match_info.get('sm','Anonymous')
        gene  = request.match_info.get('gene','Anonymous')
        seq,start,end = G[gene][0] #get only the first for now
        tiles = int(request.match_info.get('tiles','Anonymous'))
        f_p   = float(request.match_info.get('flank','Anonymous'))
        axis  = int(request.match_info.get('axis','Anonymous'))
        flank = max(0,int(round((end-start)*f_p)))
        coord = seq,max(0,start-flank),min(end+flank,ref[seq])
        meta  = query_meta(S,xs=[sample_files[sm]],sm=[sm],rg=None,seq=[seq],trk=None,ftr=None)
        W = buffer_moment_range(coord[0],coord[1],coord[2],meta,tiles=tiles,FN=8,axis=axis)
        W['coord'] = coord
    except Exception as E:
        print('Gene Symbol %s was not found:\tline 197:%s'%(gene,E))
        W = {'err':'Gene Symbol %s was not found'%gene}
    return web.json_response(W)

async def gene_track_tiles_flank_h(request):
    W = {}
    try:
        gene  = request.match_info.get('gene','Anonymous')
        track = request.match_info.get('track','Anonymous')
        seq,start,end = G[gene][0] #get only the first for now
        tiles = int(request.match_info.get('tiles','Anonymous'))
        f_p   = float(request.match_info.get('flank','Anonymous'))
        axis  = int(request.match_info.get('axis','Anonymous'))
        flank = max(0,int(round((end-start)*f_p)))
        meta = query_meta(S,xs=None,sm=None,rg=None,seq=None,trk=[track],ftr=None)
        coord = seq,max(0,start-flank),min(end+flank,ref[seq])
        W = buffer_moment_range(coord[0],coord[1],coord[2],meta,tiles=tiles,FN=8,axis=axis)
        W['coord'] = coord
    except Exception as E:
        print('Gene Symbol %s was not found:\tline 216:%s'%(gene,E))
        W = {'err':'Gene Symbol %s was not found'%gene}
    return web.json_response(W)

async def seq_start_end_tiles_flank_h(request):
    W = {}
    try:
        seq   = request.match_info.get('seq','Anonymous')
        start = int(float(request.match_info.get('start','Anonymous')))
        end   = int(float(request.match_info.get('end','Anonymous')))
        tiles = int(request.match_info.get('tiles','Anonymous'))
        f_p   = float(request.match_info.get('flank','Anonymous'))
        axis  = int(request.match_info.get('axis','Anonymous'))
        flank = max(0,int(round((end-start)*f_p)))
        coord = seq,max(0,start-flank),min(end+flank,ref[seq])
        W = buffer_moment_range(coord[0],coord[1],coord[2],S,tiles=tiles,FN=8,axis=axis)
        W['coord'] = coord
    except Exception as E:
        print('position coordinates were not found: %s:%s-%s\tline 235:%s'%(seq,start,end,E))
        W = {'err':'position coordinates were not found: %s:%s-%s'%(seq,start,end)}
    return web.json_response(W)

async def sm_trk_seq_start_end_tiles_flank_h(request):
    W = {}
    try:
        sm    = request.match_info.get('sm','Anonymous')
        trk   = request.match_info.get('trk','Anonymous')
        seq   = request.match_info.get('seq','Anonymous')
        start = int(float(request.match_info.get('start','Anonymous')))
        end   = int(float(request.match_info.get('end','Anonymous')))
        tiles = int(request.match_info.get('tiles','Anonymous'))
        f_p   = float(request.match_info.get('flank','Anonymous'))
        axis  = int(request.match_info.get('axis','Anonymous'))
        flank = max(0,int(round((end-start)*f_p)))
        coord = seq,max(0,start-flank),min(end+flank,ref[seq])
        meta  = query_meta(S,xs=[sample_files[sm]],sm=[sm],rg=None,seq=[seq],trk=[trk],ftr=None)
        W = buffer_moment_range(coord[0],coord[1],coord[2],meta,tiles=tiles,FN=8,axis=axis)
        W['coord'] = coord
    except Exception as E:
        print('position coordinates were not found: %s:%s-%s\tline 253:%s'%(seq,start,end,E))
        W = {'err':'position coordinates were not found: %s:%s-%s'%(seq,start,end)}
    return web.json_response(W)

async def sm_seq_start_end_tiles_flank_h(request):
    W = {}
    try:
        sm    = request.match_info.get('sm','Anonymous')
        seq   = request.match_info.get('seq','Anonymous')
        start = int(float(request.match_info.get('start','Anonymous')))
        end   = int(float(request.match_info.get('end','Anonymous')))
        tiles = int(request.match_info.get('tiles','Anonymous'))
        f_p   = float(request.match_info.get('flank','Anonymous'))
        axis  = int(request.match_info.get('axis','Anonymous'))
        flank = max(0,int(round((end-start)*f_p)))
        coord = seq,max(0,start-flank),min(end+flank,ref[seq])
        meta  = query_meta(S,xs=[sample_files[sm]],sm=[sm],rg=None,seq=[seq],trk=None,ftr=None)
        W = buffer_moment_range(coord[0],coord[1],coord[2],meta,tiles=tiles,FN=8,axis=axis)
        W['coord'] = coord
    except Exception as E:
        print('position coordinates were not found: %s:%s-%s\tline 274:%s'%(seq,start,end,E))
        W = {'err':'position coordinates were not found: %s:%s-%s'%(seq,start,end)}
    return web.json_response(W)

app = web.Application()
app.router.add_get('/sample_map', sample_map_h)
app.router.add_get('/gene_map', gene_map_h)
app.router.add_get('/ref_map', ref_map_h)
app.router.add_get('/genes', genes_h)
app.router.add_get('/seq/{seq}', seq_h)
app.router.add_get('/gene/{gene}', gene_h)
app.router.add_get(r'/gene/{gene}/tiles/{tiles}/flank/{flank}/axis/{axis}', gene_tiles_flank_h)
app.router.add_get('/gene/{gene}/track/{track}/tiles/{tiles}/flank/{flank}/axis/{axis}',gene_track_tiles_flank_h)
app.router.add_get('/seq/{seq}/start/{start}/end/{end}/tiles/{tiles}/flank/{flank}/axis/{axis}',seq_start_end_tiles_flank_h)
app.router.add_get('/sm/{sm}/gene/{gene}/tiles/{tiles}/flank/{flank}/axis/{axis}',sm_gene_tiles_flank_h)
app.router.add_get('/sm/{sm}/seq/{seq}/start/{start}/end/{end}/tiles/{tiles}/flank/{flank}/axis/{axis}',sm_seq_start_end_tiles_flank_h)
app.router.add_get('/sm/{sm}/trk/{trk}/seq/{seq}/start/{start}/end/{end}/tiles/{tiles}/flank/{flank}/axis/{axis}',sm_trk_seq_start_end_tiles_flank_h)
app.router.add_static('/', path='../client/', name='client')
web.run_app(app, host='127.0.0.1', port=8080)