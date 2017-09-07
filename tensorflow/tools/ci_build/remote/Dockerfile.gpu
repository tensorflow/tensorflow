FROM nvidia-cuda-clang:latest

RUN apt-get update && apt-get --no-install-recommends install -y \
    binutils \
    binutils-gold \
    curl \
    libstdc++-4.9-dev \
    python \
    python-dev \
    python-numpy \
    python-pip \
    unzip \
    zip && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Set up grpc
RUN pip install --upgrade enum34 futures mock numpy six backports.weakref autograd && \
    pip install --pre 'protobuf>=3.0.0a3' && \
    pip install 'grpcio>=1.1.3'

WORKDIR /botexec
