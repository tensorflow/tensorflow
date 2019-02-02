# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

ARG UBUNTU_VERSION=16.04

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

ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

RUN ${PIP} --no-cache-dir install \
    Pillow \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
    enum34

# Install bazel
ARG BAZEL_VERSION=0.19.2
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN ${PIP} install jupyter matplotlib
RUN ${PIP} install jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get install -y --no-install-recommends wget
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_text_classification.ipynb
COPY readme-for-jupyter.md README.md
RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /tf
EXPOSE 8888

RUN ${PYTHON} -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
