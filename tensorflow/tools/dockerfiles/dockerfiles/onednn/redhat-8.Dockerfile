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

ARG REDHAT_VERSION=latest

FROM registry.access.redhat.com/ubi8/ubi:${REDHAT_VERSION} as base

### Required OpenShift Labels
LABEL name="Intel&#174; Optimizations for TensorFlow*" \
      maintainer="Abolfazl Shahbazi <abolfazl.shahbazi@intel.com>" \
      vendor="Intel&#174; Corporation" \
      version="2.7.0" \
      release="2.7.0" \
      summary="Intel&#174; Optimizations for TensorFlow* is a binary distribution of TensorFlow* with Intel&#174; oneAPI Deep Neural Network Library primitives." \
      description="Intel&#174; Optimizations for TensorFlow* is a binary distribution of TensorFlow* with Intel&#174; oneAPI Deep Neural Network Library (Intel&#174; oneDNN) primitives, a popular performance library for deep learning applications. TensorFlow* is a widely-used machine learning framework in the deep learning arena, demanding efficient utilization of computational resources. In order to take full advantage of Intel&#174; architecture and to extract maximum performance, the TensorFlow* framework has been optimized using Intel&#174; oneDNN primitives."

# Licenses, Legal Notice and TPPs for older versions
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/LEGAL-NOTICE ./licenses/
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/LICENSE ./licenses/
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/third_party_programs_license/oneDNN-THIRD-PARTY-PROGRAMS ./licenses/third_party_programs_license/
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/third_party_programs_license/third-party-programs.txt ./licenses/third_party_programs_license/

ENV LANG C.UTF-8
ARG PYTHON=python3

### Add necessary updates here
RUN yum -y update-minimal --security --sec-severity=Important --sec-severity=Critical

RUN INSTALL_PKGS="\
    ${PYTHON}-pip \
    which" && \
    yum -y --setopt=tsflags=nodocs install $INSTALL_PKGS && \
    rpm -V $INSTALL_PKGS && \
    yum -y clean all --enablerepo='*'

# Intel Optimizations specific Envs
ENV KMP_AFFINITY='granularity=fine,verbose,compact,1,0' \
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PYTHON=python3

RUN yum update -y && yum install -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    which && \
    yum clean all


RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python

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

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
