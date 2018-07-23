# Tensorflow ROCm port: Basic installation

## Intro
This instruction provides a starting point for TensorFlow ROCm port (mostly via deb packages).
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install ROCm
```
export ROCM_PATH=/opt/rocm
export DEBIAN_FRONTEND noninteractive
sudo apt update && sudo apt install -y wget software-properties-common 
```

Add the ROCm repository:  
```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
```
Install misc pkgs:
```
sudo apt-get update && sudo apt-get install -y \
  build-essential \
  clang-3.8 \
  clang-format-3.8 \
  clang-tidy-3.8 \
  cmake \
  cmake-qt-gui \
  ssh \
  curl \
  apt-utils \
  pkg-config \
  g++-multilib \
  git \
  libunwind-dev \
  libfftw3-dev \
  libelf-dev \
  libncurses5-dev \
  libpthread-stubs0-dev \
  vim \
  gfortran \
  libboost-program-options-dev \
  libssl-dev \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev \
  rpm \
  wget && \
  sudo apt-get clean && \
  sudo rm -rf /var/lib/apt/lists/*
```

Install ROCm pkgs:
```
sudo apt-get update && \
    sudo apt-get install -y --allow-unauthenticated \
    rocm-dkms rocm-dev rocm-libs \
    rocm-device-libs \
    hsa-ext-rocr-dev hsakmt-roct-dev hsa-rocr-dev \
    rocm-opencl rocm-opencl-dev \
    rocm-utils \
    rocm-profiler cxlactivitylogger \
    miopen-hip miopengemm
```

Add username to 'video' group and reboot:  
```
sudo adduser $LOGNAME video
sudo reboot
```

## Install required python packages

On Python 2-based systems:
```
sudo apt-get update && sudo apt-get install -y \
    python-numpy \
    python-dev \
    python-wheel \
    python-mock \
    python-future \
    python-pip \
    python-yaml \
    python-setuptools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
```

On Python 3-based systems:
```
sudo apt-get update && sudo apt-get install -y \
    python3-numpy \
    python3-dev \
    python3-wheel \
    python3-mock \
    python3-future \
    python3-pip \
    python3-yaml \
    python3-setuptools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
```

## Install TensorFlow ROCm port

Uninstall any previously-installed tensorflow whl packages:  
```
pip list | grep tensorflow && pip uninstall -y tensorflow
```

For Python 2-based systems:
```
# wget the TensorFlow ROCm port whl package
cd ~/ && wget http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp27-cp27mu-manylinux1_x86_64.whl

# Pip install the whl package 
cd ~ && sudo pip install tensorflow-1.8.0-cp27-cp27mu-manylinux1_x86_64.whl && rm tensorflow-1.8.0-cp27-cp27mu-manylinux1_x86_64.whl
```

For Python 3-based systems:
```
# wget the TensorFlow ROCm port whl package
cd ~/ && wget http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp35-cp35m-manylinux1_x86_64.whl

# Pip3 install the whl package 
cd ~ && sudo pip3 install tensorflow-1.8.0-cp35-cp35m-manylinux1_x86_64.whl && rm tensorflow-1.8.0-cp35-cp35m-manylinux1_x86_64.whl
```
