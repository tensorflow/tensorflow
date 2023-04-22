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

ARG UBUNTU_VERSION=18.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.2.2-1
ARG LIBNVINFER_MAJOR_VERSION=7

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        libcudnn8=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
# NOTE: libnvinfer uses cuda11.1 versions
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.1 \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.1 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
ARG TF_PACKAGE=tensorflow
RUN apt-get update && apt-get install -y curl libhdf5-dev wget
RUN python3 -m pip install --no-cache-dir --global-option=build_ext \
            --global-option=-I/usr/include/hdf5/serial/ \
            --global-option=-L/usr/lib/powerpc64le-linux-gnu/hdf5/serial \
            h5py

# CACHE_STOP is used to rerun future commands, otherwise downloading the .whl will be cached and will not pull the most recent version
ARG CACHE_STOP=1
RUN if [ ${TF_PACKAGE} = tensorflow-gpu ]; then \
        BASE=https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/lastSuccessfulBuild/; \
    elif [ ${TF_PACKAGE} = tf-nightly-gpu ]; then \
        BASE=https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Nightly_Artifact/lastSuccessfulBuild/; \
    elif [ ${TF_PACKAGE} = tensorflow ]; then \
        BASE=https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/lastSuccessfulBuild/; \
    elif [ ${TF_PACKAGE} = tf-nightly ]; then \
        BASE=https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Nightly_Artifact/lastSuccessfulBuild/; \
    fi; \
    MAJOR=`python3 -c 'import sys; print(sys.version_info[0])'`; \
    MINOR=`python3 -c 'import sys; print(sys.version_info[1])'`; \
    PACKAGE=$(wget -qO- ${BASE}"api/xml?xpath=//fileName&wrapper=artifacts" | grep -o "[^<>]*cp${MAJOR}${MINOR}[^<>]*.whl"); \
    wget ${BASE}"artifact/tensorflow_pkg/"${PACKAGE}; \
    python3 -m pip install --no-cache-dir ${PACKAGE}

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
