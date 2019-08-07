FROM ubuntu:bionic
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
build-essential \
g++ \
gfortran \
git \
wget \
nano \
libffi6 \
libffi-dev \
libssl1.0.0 \
libssl-dev \
libblas3 \
libblas-dev \
liblapack3 \
liblapack-dev \
libcurl4-openssl-dev \
libxml2-dev \
libncurses5-dev \
libbz2-dev \
zlib1g-dev \
python \
python-dev \
python-pip
RUN apt-get autoremove
RUN pip install subprocess32
RUN pip install -Iv 'Cython>=0.29.0,<0.30.0'
RUN pip install -Iv 'numpy>=1.16.0,<1.17.0'
RUN pip install -Iv 'h5py>=2.9.0,<3.0.0'
RUN pip install -Iv 'pysam>=0.15.0,<0.16.0'
RUN pip install https://github.com/timothyjamesbecker/hfm/releases/download/0.1.1/hfm-0.1.1.tar.gz
