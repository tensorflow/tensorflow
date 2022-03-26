# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as base

# Intel Optimizations specific Envs
ENV KMP_AFFINITY='granularity=fine,verbose,compact,1,0' \
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PYTHON=python3

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    curl \
    ${PYTHON} \
    ${PYTHON}-distutils

RUN curl -fSsL https://bootstrap.pypa.io/get-pip.py | ${PYTHON}

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

ARG DEBIAN_FRONTEND="noninteractive"

# install mpich, openssh for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    mpich \
    libmpich-dev \
    openssh-client \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a wrapper for MPICH to allow running as root by default
RUN mv /usr/bin/mpirun /usr/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/bin/mpirun && \
    echo 'mpirun.real "$@"' >> /usr/bin/mpirun && \
    chmod a+x /usr/bin/mpirun

# Set up SSH
RUN mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install Horovod
ARG HOROVOD_WITHOUT_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=v0.21.1

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    cmake \
    g++-8 \
    gcc-8 \
    git \
    ${PYTHON}-dev

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8

RUN ${PYTHON} -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
