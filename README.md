# hfm|s|t
Exact Hierarchical Feature Moment/Spectrum/Transition Extraction<br>

#####Python 2.7.15+ Python PIP module or stand alone python commandline tool

#### Input Overview:
CRAM/BAM/SAM file=> in.bam<br><br>
Sequence Alignment Map or compressed file forms (SAM,BAM,CRAM) <br>
#### Output Overview:
HDF5 file=> out.hdf5<br><br>
HDF5 container that provides hierarchical summarized features over SAM flags and alignment properties, counts and other important attributes<br> The resultingdata store can be used on any platform and loaded into C,C++,Java,R or node.js envronments.
The python library offers additional functionality via the HFM class that provides buffering, transformation and functionality for open data anlaysis workflows. Buffering of the avaible window levels are done in chucks and looks like a map/dictionary that provides an array payload (flat multiprocessing Array or numpy data array for machine-learning use are supported)<br>


#### TRACKS
#####Features over which moments, spectrum or transistions can be extracted:<br>
(1) <b>total</b> = all reads aligned at the position<br>
(2) <b>proper_pair</b> = reads aligned at the position with the proper pairing bit set<br>
(3) <b>discordant</b> = reads that are not properly paired but map to the same seq<br>
(4) <b>primary</b> = reads aligned at the position that have the primary bit set meaning its the best mapping<br>
(5) <b>alternate</b> = reads aligned at the position that are either supplimentary or secondary (not primary)<br>
(6) <b>orient_same</b> = reads aligned at the position are in the same direction (-> -> or <- <-)<br>
(7) <b>orient_out</b> = reads aligned at the position are in the outward direction (<- ->) <br>
(8) <b>orient_um</b> = reads aligned at the position have an unmapped mate -> * or * <-<br>
(9) <b>orient_chr</b> = reads aligned at the position have a mate that is mapped to another seq<br>
(10) <b>clipped</b> = reads aligned at the position have either a softclip or hardclip in the CIGAR meaning that only part of the read maps<br>
(11) <b>deletion</b> = reads aligned at the position have a deletion inside detailed by the CIGAR<br>
(12) <b>insertion</b> = reads aligned at the position have an insertion inside detailed by the CIGAR<br>
(13) <b>substitution</b> = reads aligned at the position have a substitution inside detailed by the CIGAR<br>
(14) <b>fwd_rev_diff</b> = the difference of -> to <- used for strand bias (RNA-seq)<br>
(15) <b>mapq></b> = average mapping quality of all reads aligned at the position<br>
(16) <b>mapq_pp</b> = average mapping quality of the properly paired reads aligned at the position<br>
(17) <b>mapq_dis</b> = average mapping quality of the discordant reads aligned at the position<br>
(18) <b>tlen</b> = average insert length of all paired reads aligned at the position<br>
(19) <b>tlen_pp</b> = average insert length of the properly paired reads aligned at the position<br>
(20) <b>tlen_dis</b> = the average insert length of the discordant reads aligned at the position<br>
(21) <b>GC</b> = the average GC content (fast estimate) of the reads aligned at the position<br>
<br>
For each track we summarize using a base window. A base window is a starting point.  For disjoint window hierarchies, we use a numerically stable exact calculation of moments (Pebay, Philippe Pierre. Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments. United States: N. p., 2008. Web. doi:10.2172/1028931) And for sliding windows we use an approximate variation that restarts extact calculations every x number of calculations where x is a fraction of the window size.
#### FEATURES 
##### (either tiling which is disjoint window summaries of each track or sliding which is a summary window of size w that is moved across the track by one position each time):
(1) <b>moments</b> = quick summaries that include: [1] window size [2] sum [3] min [4] max [5] M1 [6] M2 [7] M3 [8] M4 where M1-M4 are the first four raw moments. included are functions to create centered normalized statistics such as the standard deviation, kurtosis and skewness.<br>
(2) <b>spectrum</b> = a histogram (1D) of the values contained inside (works best for count tracks such as total, proper_pair, ect) but the use can supply an array of numerical boundaries to use (future work could be random-sampling estimation for auto setting this.<br>
(3) <b>transitions</b> = a 2D array of the values that change to other values at the next position.  This can be used to find the position at which tracks are missing and also to gneral describe the shape of the singal in the tracks windows.<br>

### INSTALLATION
requires python 2.7.15+ (untested with python 3) and packages: numpy 1.16.0+, h5py 2.9.0+, pysam 0.15.2<br>
core.pyx is a cython/pyrex file that when compiled will generate C code and a .so (shared object)<br>
This is for high performance reading of the CARM/BAM/SAM input, which is the slowest section of extraction (when profiled)<br>
The following installation will build and install the safe library into your python distribution<br>

```bash
pip install https:/github.com/timothyjamesbecker/hfm.tar.gz
```
### DOCKER IMAGE
alternative: download and install docker toolbox and then (avaible soon):
```bash
docker pull timothyjamesbecker/hfm
sudo docker run -v /mydata:/data -it timothyjamesbecker/hfm extractor.py -h
```
### Python hfm module Tutorial/Examples
read the total chr1 alignments for read group SRR622461 for file NA12878.dna.bam and generate moments, spectrum and transition features
```python
import hfm
s = hfm.HFM(window=100,window_branch=10,window_root=int(1E9))
s.extract_seq('NA12878.dna.bam','NA12878',sms={'NA12878':'SRR622461'},seq={'chr1':249250621},
              merge_rg=True,tracks=['total'],features=['all'])
#reads the chr1 reads for SRR622461 read groups for file NA12878.dna.bam
```

### HFM class details
```python

```
