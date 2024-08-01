#!/bin/bash

# Set CUDA and cuDNN paths
export CUDA_HOME=/usr/local/cuda-12.2
export CUDNN_INCLUDE_DIR=/usr/local/cuda-12.2/include
export CUDNN_LIB_DIR=/usr/local/cuda-12.2/lib64

# Ensure CUDA compiler is in PATH
export PATH=$CUDA_HOME/bin:$PATH

# Verify CUDA compiler
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Please ensure CUDA is installed and nvcc is in your PATH."
    exit 1
fi

# Check CUDA version
nvcc --version

# Verify cuDNN installation
if [ ! -f $CUDNN_INCLUDE_DIR/cudnn.h ]; then
    echo "cuDNN header file not found. Please ensure cuDNN is installed correctly."
    exit 1
fi

if [ ! -f $CUDNN_LIB_DIR/libcudnn.so ]; then
    echo "cuDNN library file not found. Please ensure cuDNN is installed correctly."
    exit 1
fi

echo "CUDA and cuDNN are set up correctly."
