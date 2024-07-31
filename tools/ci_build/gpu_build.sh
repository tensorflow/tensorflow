#!/bin/bash

# Define CUDA and cuDNN paths
export CUDA_HOME=/usr/local/cuda-12.2
export CUDNN_INCLUDE_DIR=/usr/local/cuda-12.2-cudnn/include
export CUDNN_LIB_DIR=/usr/local/cuda-12.2-cudnn/lib64

# Ensure CUDA compiler is in PATH
export PATH=$CUDA_HOME/bin:$PATH

# Verify CUDA compiler
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Please ensure CUDA is installed and nvcc is in your PATH."
    exit 1
fi

# Check CUDA version
nvcc --version
