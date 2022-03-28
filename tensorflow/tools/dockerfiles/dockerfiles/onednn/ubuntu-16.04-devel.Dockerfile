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

FROM ubuntu:${UBUNTU_VERSION} AS base

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        sudo \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CI_BUILD_PYTHON python

# CACHE_STOP is used to rerun future commands, otherwise cloning tensorflow will be cached and will not pull the most recent version
ARG CACHE_STOP=1
# Check out TensorFlow source code if --build-arg CHECKOUT_TF_SRC=1
ARG CHECKOUT_TF_SRC=0
ARG TF_BRANCH=master
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git --branch "${TF_BRANCH}" --single-branch /tensorflow_src || true

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PYTHON=python3

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON}

RUN curl -fSsL https://bootstrap.pypa.io/get-pip.py | ${PYTHON}

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl

# Installs bazelisk
RUN mkdir /bazel && \
    curl -fSsL -o /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    mkdir /bazelisk && \
    curl -fSsL -o /bazelisk/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazelisk/master/LICENSE" && \
    mkdir -p "$HOME/bin" && \
    curl -fSsL -o $HOME/bin/bazel "https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64" && \
    chmod u+x "$HOME/bin/bazel" && \
    export PATH="$HOME/bin:$PATH"

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
