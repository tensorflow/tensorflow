# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /tf_wheel /tf_builder

WORKDIR /tf_builder

RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    curl \
    git \
    wget \
    patchelf \
    rsync \
    zip \
    unzip \
    build-essential \
    dos2unix \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Clang-17 using llvm.sh (while system python is intact)
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 17 && \
    rm llvm.sh && \
    ln -sf /usr/bin/clang-17 /usr/bin/clang && \
    ln -sf /usr/bin/clang++-17 /usr/bin/clang++

# Python-3.11 setup
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set default python3 to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Bazelisk
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 && \
    chmod +x bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Python build dependencies
RUN pip install -U pip setuptools wheel packaging requests opt_einsum numpy keras_preprocessing

COPY build_tf_noavx.sh /tf_builder/
RUN sed -i 's/\r$//' /tf_builder/build_tf_noavx.sh && \
    chmod +x /tf_builder/build_tf_noavx.sh

CMD ["/tf_builder/build_tf_noavx.sh"]
