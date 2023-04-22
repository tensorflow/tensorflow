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

ARG CENTOS_VERSION=8

FROM centos:${CENTOS_VERSION} AS base

# Enable both PowerTools and EPEL otherwise some packages like hdf5-devel fail to install
RUN yum clean all && \
    yum update -y && \
    yum install -y epel-release

RUN yum update -y && \
    yum install -y \
        curl \
        freetype-devel \
        gcc \
        gcc-c++ \
        git \
        hdf5-devel \
        java-1.8.0-openjdk \
        java-1.8.0-openjdk-devel \
        java-1.8.0-openjdk-headless \
        libcurl-devel \
        make \
        pkg-config \
        rsync \
        sudo \
        unzip \
        zeromq-devel \
        zip \
        zlib-devel && \
        yum clean all

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

# On CentOS 7, yum needs to run with Python2.7
RUN sed -i 's#/usr/bin/python#/usr/bin/python2#g' /usr/bin/yum /usr/libexec/urlgrabber-ext-down

# Install bazel
ARG BAZEL_VERSION=3.7.2
RUN mkdir /bazel && \
    curl -fSsL -o /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    curl -fSsL -o /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    bash /bazel/installer.sh && \
    rm -f /bazel/installer.sh

RUN yum update -y && yum install -y \
    openmpi \
    openmpi-devel \
    openssh \
    openssh-server \
    which && \
    yum clean all

ENV PATH="/usr/lib64/openmpi/bin:${PATH}"

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv -f $(which mpirun) /usr/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/bin/mpirun && \
    chmod a+x /usr/bin/mpirun

# Configure OpenMPI to run good defaults:
RUN echo "btl_tcp_if_exclude = lo,docker0" >> /etc/openmpi-x86_64/openmpi-mca-params.conf

# Install OpenSSH for MPI to communicate between containers
RUN mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/sshd_config | grep -v StrictHostKeyChecking > /etc/ssh/sshd_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/sshd_config.new && \
    mv -f /etc/ssh/sshd_config.new /etc/ssh/sshd_config

# Check out horovod source code if --build-arg CHECKOUT_HOROVOD_SRC=1
ARG CHECKOUT_HOROVOD_SRC=0
ARG HOROVOD_BRANCH=master
RUN test "${CHECKOUT_HOROVOD_SRC}" -eq 1 && git clone --branch "${HOROVOD_BRANCH}" --single-branch --recursive https://github.com/uber/horovod.git /horovod_src || true

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN ${PYTHON} -m pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
RUN ${PYTHON} -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir -p /tf/ && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /tf
EXPOSE 8888

RUN ${PYTHON} -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
