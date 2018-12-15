FROM nvidia/cuda:10.0-base-ubuntu${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cudart-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libcudnn7-dev=7.4.1.5-1+cuda10.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        && \
    find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0 \
        && rm -rf /var/lib/apt/lists/*

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
ENV TF_CUDA_VERSION=10.0
ENV TF_CUDNN_VERSION=7

# Check out TensorFlow source code if --build_arg CHECKOUT_TENSORFLOW=1
ARG CHECKOUT_TF_SRC=0
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src
