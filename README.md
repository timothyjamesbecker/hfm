![GitHub All Releases](https://img.shields.io/github/downloads/timothyjamesbecker/hfm/total.svg) [![DOI](https://zenodo.org/badge/192426986.svg)](https://zenodo.org/badge/latestdoi/192426986) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)<br>
![Alt text](images/logo.png?raw=true "hfm")
### A Hierarchical Feature Moment/Spectrum/Transition Framework for Analyzing and Visualizing Large Cohort Omic Data<br>
Copyright (C) 2019-2020 Timothy James Becker<br>
```
Becker, T. and Shin, DG.(2020) "Efficient methods for hierarchical multi-omic feature extraction and
visualization", Int. J. Data Mining and Bioinformatics Vol 23, No. 4, pp 285-298.
```
```bash
T. J. Becker and D. Shin, "HFM: Hierarchical Feature Moment Extraction for Multi-Omic Data Visualization,"
2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), San Diego, CA, USA, 2019, pp. 1970-1976.
```
##### Python 2.7.15+ Python PIP module or python commandline tool

#### PIP INSTALLATION AND TESTING
requires python 2.7.15+ (untested with python 3) and packages: numpy 1.16.0+, h5py 2.9.0+, pysam 0.15.2, subproccess32 (for extractor.py usage)<br>
core.pyx is a cython/pyrex file that when compiled will generate C code and a .so (shared object)<br>
This is for high performance reading and statistical summary of the CRAM/BAM/SAM input, which is the slowest section of extraction<br>
The following installation will build and install the hfm library into your python distribution<br>

```bash
pip install https://github.com/timothyjamesbecker/hfm/releases/download/0.1.2/hfm-0.1.2.tar.gz
extractor.py --test -o ~/hfm_test/ -v total -s 1

```
#### DOCKER IMAGE DOWNLOAD AND TESTING
alternative: download and install docker toolbox and then (now avaible):
```bash
docker pull timothyjamesbecker/hfm
docker run -v ~/:/data -it timothyjamesbecker/hfm extractor.py --test -o /data/hfm_test/ -v total -s 1
```

#### Input Overview:
CRAM/BAM/SAM file=> in.bam<br><br>
Sequence Alignment Map or compressed file forms (SAM,BAM,CRAM)<br>
We read all SM:RG tags and map them so that multi-sample multi-read groups are in tact, they are merged by default in the extractor.py command line tool however.
#### Output Overview:
HDF5 file=> out.hdf5<br><br>
HDF5 container that provides hierarchical summarized features over SAM flags and alignment properties, counts and other important attributes<br> The resultingdata store can be used on any platform and loaded into C,C++,Java,R or node.js envronments.
The python library offers additional functionality via the HFM class that provides buffering, transformation and functionality for open data anlaysis workflows. Buffering of the avaible window levels are done in chucks and looks like a map/dictionary that provides an array payload (flat multiprocessing Array or numpy data array for machine-learning use are supported)<br>


#### HFM extractor.py command line tool
Once installation is complete, you will have access in your environment to the command line tool that will extract either single or multi SAM/BAM/CRAM files.
```
usage: extractor.py [-h] [-i IN_PATH] [-r REF_PATH] [-o OUT_DIR] [-p CPUS]
                    [-c COMP] [-n] [-w WINDOW] [-b BRANCH] [-t TILE] [-s SEQS]
                    [-v VECTORS] [-f FEATURES] [--test]
                    [--reproc_dir REPROC_DIR]

HFM: Exact Hierarchical Feature Moment/Spectrum/Transition
Extraction for Analysis and Visualization
Batch Extractor Tool 0.1.2, Copyright (C) 2019 Timothy James Becker

optional arguments:
  -h, --help            show this help message and exit
  -i IN_PATH, --in_path IN_PATH
                        sam/bam/cram file or input directory	[None]
  -r REF_PATH, --ref_path REF_PATH
                        for cram inputs	[None]
  -o OUT_DIR, --out_dir OUT_DIR
                        output directory	[None]
  -p CPUS, --cpus CPUS  number of parallel core||readers (pools to sequences)	[1]
  -c COMP, --comp COMP  compression type	[lzf or gzip]
  -n, --no_merge_rg     do not merge all rg into one called "all"	[False]
  -w WINDOW, --window WINDOW
                        window size in bp	[100]
  -b BRANCH, --branch BRANCH
                        window branching factor	[10]
  -t TILE, --tile TILE  use tiles for features as opposed to 1-bp sliding windows of size w	[True]
  -s SEQS, --seqs SEQS  comma seperated list of seqs that will be extracted, 	[all]
  -v VECTORS, --vectors VECTORS
                        comma seperated list of vectors that will be extracted for each seq, all gives every available	[total]
  -f FEATURES, --features FEATURES
                        comma seperated list of features that will be calculated for each vector on each sequence, all gives every available	[moments]
  --test                will run the multisample.bam test file and save result in the out_dir
  --reproc_dir REPROC_DIR
                        output directory for rebranching and retransforming hfm data from base windows	[None]

```

##### TRACKS (VECTORS)
##### Features over which moments, spectrum or transistions can be extracted:<br>
(1) <b>total</b> = all reads aligned at the position<br>
(2) <b>proper_pair</b> = reads aligned at the position with the proper pairing bit set<br>
(3) <b>discordant</b> = reads that are not properly paired but map to the same seq<br>
(4) <b>primary</b> = reads aligned at the position that have the primary bit set meaning its the best mapping<br>
(5) <b>alternate</b> = reads aligned at the position that are either supplimentary or secondary (not primary)<br>
(6) <b>orient_same</b> = reads aligned at the position are in the same direction (-> -> or <- <-)<br>
(7) <b>orient_out</b> = reads aligned at the position are in the outward direction (<- ->) <br>
(8) <b>orient_um</b> = reads aligned at the position have an unmapped mate -> * or * <-<br>
(9) <b>orient_chr</b> = reads aligned at the position have a mate that is mapped to another seq<br>
(10) <b>clipped</b> = reads aligned at the position have either a softclip or hardclip in the CIGAR<br>
(11) <b>left clipped</b> = reads aligned at the position have either a left softclip or hardclip in the CIGAR<br>
(12) <b>right clipped</b> = reads aligned at the position have either a right softclip or hardclip in the CIGAR<br>
(13) <b>deletion</b> = reads aligned at the position have a deletion inside detailed by the CIGAR<br>
(14) <b>insertion</b> = reads aligned at the position have an insertion inside detailed by the CIGAR<br>
(15) <b>substitution</b> = reads aligned at the position have a substitution inside detailed by the CIGAR<br>
(16) <b>fwd_rev_diff</b> = the difference of -> to <- used for strand bias (RNA-seq)<br>
(17) <b>mapq</b> = average mapping quality of all reads aligned at the position<br>
(18) <b>mapq_pp</b> = average mapping quality of the properly paired reads aligned at the position<br>
(19) <b>mapq_dis</b> = average mapping quality of the discordant reads aligned at the position<br>
(20) <b>tlen</b> = average insert length of all paired reads aligned at the position<br>
(21) <b>tlen_pp</b> = average insert length of the properly paired reads aligned at the position<br>
(22) <b>tlen_dis</b> = the average insert length of the discordant reads aligned at the position<br>
(23) <b>GC</b> = the average GC content (fast estimate) of the reads aligned at the position<br>
<br>
For each track we summarize using a base window. A base window is a starting point.  For disjoint window hierarchies, we use a numerically stable exact calculation of moments (Pebay, Philippe Pierre. Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments. United States: N. p., 2008. Web. doi:10.2172/1028931) And for sliding windows we use an approximate variation that restarts extact calculations every x number of calculations where x is a fraction of the window size.
#### FEATURES 
##### (either tiling which is disjoint window summaries of each track or sliding which is a summary window of size w that is moved across the track by one position each time):
(1) <b>moments</b> = quick summaries that include: [1] window size [2] sum [3] min [4] max [5] M1 [6] M2 [7] M3 [8] M4 where M1-M4 are the first four raw moments. included are functions to create centered normalized statistics such as the standard deviation, kurtosis and skewness.<br>
(2) <b>spectrum</b> = a histogram (1D) of the values contained inside (works best for count tracks such as total, proper_pair, ect) but the use can supply an array of numerical boundaries to use (future work could be random-sampling estimation for auto setting this.<br>
(3) <b>transitions</b> = a 2D array of the values that change to other values at the next position.  This can be used to find the position at which tracks are missing and also to gneral describe the shape of the singal in the tracks windows.<br>

### Python hfm module Tutorial/Examples
read the total chr1 alignments for read group SRR622461 for file NA12878.dna.bam and generate moments, spectrum and transition features
```python
from hfm import hfm
s = hfm.HFM(window=100,window_branch=10,window_root=int(1E9))
s.extract_seq('NA12878.dna.bam','NA12878',sms={'NA12878':'SRR622461'},seq={'chr1':249250621},
              merge_rg=True,tracks=['total'],features=['moments'])
#reads the chr1 reads for SRR622461 read groups for file NA12878.dna.bam
```
