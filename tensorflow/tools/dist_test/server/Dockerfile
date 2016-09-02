# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# Test server for TensorFlow GRPC server
#
# To build the image, use ../build_server.sh

FROM ubuntu:14.04

MAINTAINER Shanqing Cai <cais@google.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
        curl \
        python-numpy \
        python-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow CPU version from nightly build
RUN pip --no-cache-dir install \
    http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.10.0-cp27-none-linux_x86_64.whl

# Copy files, including the GRPC server binary at
# server/grpc_tensorflow_server.py
ADD . /var/tf-k8s

# Container entry point
ENTRYPOINT ["/var/tf-k8s/server/grpc_tensorflow_server.py"]
