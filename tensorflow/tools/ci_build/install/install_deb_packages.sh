#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

set -e

# Install dependencies from ubuntu deb repository.
apt-get update

# gfortran, atlas, blas and lapack required by scipy pip install
apt-get install -y \
    autoconf \
    automake \
    bc \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    gfortran \
    libatlas-base-dev \
    libblas-dev \
    libcurl4-openssl-dev \
    liblapack-dev \
    libtool \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    pkg-config \
    python-dev \
    python-numpy \
    python-pip \
    python-virtualenv \
    python3-dev \
    python3-numpy \
    python3-pip \
    sudo \
    swig \
    unzip \
    wget \
    zip \
    zlib1g-dev
apt-get clean
rm -rf /var/lib/apt/lists/*
