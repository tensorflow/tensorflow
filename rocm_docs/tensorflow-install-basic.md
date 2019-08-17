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
  clang \
  clang-format \
  clang-tidy \
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
    rocm-dkms rocm-dev rocm-libs rccl \
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
## Install Tensorflow Community Supported Builds 
Link to the upstream Tensorflow CSB doc:
https://github.com/tensorflow/tensorflow#community-supported-builds

We provide nightly tensorflow-rocm whl packages for Python 2.7, 3.5, 3.6 and 3.7 based systems.
After downloading the compatible whl package, you can use pip/pip3 to install.

For example, the following commands can be used to download and install the tensorflow-rocm CSB package on an Ubuntu 16.04 system previously configured with ROCm and Python3.5:
```
wget http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/lastSuccessfulBuild/artifact/pip35_test/whl/tensorflow_rocm-1.14.0-cp35-cp35m-manylinux1_x86_64.whl
pip3 install --user tensorflow_rocm-1.14.0-cp35-cp35m-manylinux1_x86_64.whl
```

## Install TensorFlow ROCm release build

Uninstall any previously-installed tensorflow whl packages:  
```
pip list | grep tensorflow && pip uninstall -y tensorflow
```

We maintain `tensorflow-rocm` whl packages on PyPI [here](https://pypi.org/project/tensorflow-rocm).

For Python 2-based systems:
```
# Pip install the whl package from PyPI
pip install --user tensorflow-rocm --upgrade
```

For Python 3-based systems:
```
# Pip3 install the whl package from PyPI
pip3 install --user tensorflow-rocm --upgrade
```
