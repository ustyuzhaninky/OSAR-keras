FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# FROM tensorflow/tensorflow:latest-gpu-jupyter

LABEL maintainer="Konstantin Ustyuzhanin <ustyuzhaninky@hotmail.com>"
LABEL org.jupyter.service="osar-base"
LABEL version="1.0"
ENV buildTag='latest'
ENV LANG=en_US.UTF-8

# add a cleaner
ADD res/clean-layer.sh /tmp/clean-layer.sh

ENV DEBIAN_FRONTEND noninteractive

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --fix-missing --no-install-recommends apt-utils \
        build-essential \
        curl \
	binutils \
	gdb \
        git \
	freeglut3 \
	freeglut3-dev \
	libxi-dev \
	libxmu-dev \
	gfortran \
        pkg-config \
	python-numpy \
	python-dev \
	python-setuptools \
	libboost-python-dev \
	libboost-thread-dev \
        pbzip2 \
        rsync \
        software-properties-common \
        libboost-all-dev \
        libopenblas-dev \ 
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
	libgraphicsmagick1-dev \
        libavresample-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev \
	libgraphicsmagick1-dev \
	libavcodec-dev \
	libgtk2.0-dev \
	liblapack-dev \
        liblapacke-dev \
	libswscale-dev \
	libcanberra-gtk-module \
        libboost-dev \
	libboost-all-dev \
        libeigen3-dev \
	wget \
        vim \
        qt5-default \
        unzip \
	zip \ 
        && \
    /tmp/clean-layer.sh

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get install -y cmake make g++ && \
    /tmp/clean-layer.sh

RUN apt-get update && \
    /tmp/clean-layer.sh

ARG PYTHON=python3
ARG VER=8
ARG PIP=pip3

# installing python 3.8
RUN apt-get update && apt-get install -y \
    ${PYTHON}.${VER} \
    ${PYTHON}-pip && \
    /tmp/clean-layer.sh

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools \
    hdf5storage \
    h5py \
    py3nvml \
    scikit-image \
    scikit-learn \
    matplotlib \
    pyinstrument && \
    /tmp/clean-layer.sh

WORKDIR /

# dlib
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.16' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA --clean && \
    /tmp/clean-layer.sh

# set locales and upgrade everything
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y locales && \
    rm -rf /var/lib/apt/lists/* && \
    /tmp/clean-layer.sh

# Install TensorRT (TPU Access)
# RUN apt-get update && \
#     apt-get install -y nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64 && \
#     apt-get update && \
#     apt-get install -y libnccl2=2.8.3-1+cuda11.2 && \
#     /tmp/clean-layer.sh

RUN file="$(ls -1 /usr/local/)" && echo $file

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PIP_DEFAULT_TIMEOUT 180

# set work directory
WORKDIR /usr/src/

# copy package files
COPY ./requirements.txt /usr/src/requirements.txt
COPY ./OSAR /usr/src/OSAR
COPY ./osar_version.py /usr/src/osar_version.py
COPY ./setup.py /usr/src/setup.py
COPY ./LICENSE /usr/src/LICENSE
COPY ./README.md /usr/src/README.md

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    /tmp/clean-layer.sh

# install the package
RUN pip install /usr/src/. && \
    /tmp/clean-layer.sh

# install analytics for further use
RUN pip install seaborn
RUN pip install numba

# install jupyter notebook
RUN pip install jupyter-tabnine
RUN pip install cupy-cuda112
RUN pip install mlflow 
RUN pip install seldon-core 
RUN pip install albumentations 
RUN pip install networkx 
RUN pip install jupyter-tabnine 
RUN pip install shap 
RUN pip install tensor-sensor 
RUN pip install fastapi

WORKDIR //tf/notebooks
COPY colab/. //tf/notebooks
COPY ./README.md //tf/notebooks/README.md

EXPOSE 8888

# installing atari ROMs for use with atari_py, gym and pybullet
RUN pip install atari_py
RUN add-apt-repository multiverse && apt-get update && apt-get install -y unrar
RUN wget http://www.atarimania.com/roms/Roms.rar && \
    unrar x Roms.rar && rm -f Roms.rar && \
    unzip ROMS && rm -f ROMS.zip && \
    unzip "HC ROMS" && rm -f "HC ROMS.zip" && \
    python3 -m atari_py.import_roms ROMS && \
    python3 -m atari_py.import_roms HC_ROMS && \
    rm -r ROMS && rm -r "HC ROMS" && \
    /tmp/clean-layer.sh

# Better container security versus running as root
RUN useradd -ms /bin/bash container_user

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

