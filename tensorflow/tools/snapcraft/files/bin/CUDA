#!/bin/bash

# install CUDA Toolkit v10.1
CUDA_REPO_PKG="cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/${CUDA_REPO_PKG}
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-get update
sudo apt-get -y install cuda

# install cuDNN v7.6
CUDNN_TAR_FILE="cudnn-10.1-linux-x64-v7.6.0.64.tgz"
wget https://developer.download.nvidia.com/compute/redist/cudnn/v7.6/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-10.1/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64/
sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*

# set environment variables
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
