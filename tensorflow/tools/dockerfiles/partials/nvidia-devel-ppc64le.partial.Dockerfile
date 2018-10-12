ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda-ppc64le:9.2-base-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-2 \
        cuda-cublas-dev-9-2 \
        cuda-cudart-dev-9-2 \
        cuda-cufft-dev-9-2 \
        cuda-curand-dev-9-2 \
        cuda-cusolver-dev-9-2 \
        cuda-cusparse-dev-9-2 \
        curl \
        git \
        libcudnn7=7.2.1.38-1+cuda9.2 \
        libcudnn7-dev=7.2.1.38-1+cuda9.2 \
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
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.2/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/powerpc64le-linux-gnu/libcudnn_static_v7.a
