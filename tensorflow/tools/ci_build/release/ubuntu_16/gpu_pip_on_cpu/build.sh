#!/bin/bash
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
# ==============================================================================
set -e
set -x

source tensorflow/tools/ci_build/release/common.sh
update_bazel_linux

install_ubuntu_16_pip_deps pip3.6

# Run configure.
export TF_NEED_GCP=1
export TF_NEED_HDFS=1
export TF_NEED_S3=1
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=10
export TF_CUDNN_VERSION=7
export TF_NEED_TENSORRT=1
export TENSORRT_INSTALL_PATH=/usr/local/tensorrt
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=$(which python3.6)
export LD_LIBRARY_PATH="/usr/local/cuda:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$TENSORRT_INSTALL_PATH/lib"
export TF_CUDA_COMPUTE_CAPABILITIES=3.5,3.7,5.2,6.0,6.1,7.0

yes "" | "$PYTHON_BIN_PATH" configure.py

########################
## Build GPU pip package
########################
bazel build --config=opt \
  --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain \
  tensorflow/tools/pip_package:build_pip_package

PIP_WHL_DIR=whl
mkdir -p ${PIP_WHL_DIR}
PIP_WHL_DIR=$(readlink -f ${PIP_WHL_DIR})  # Get absolute path
bazel-bin/tensorflow/tools/pip_package/build_pip_package "${PIP_WHL_DIR}"
WHL_PATH=$(ls "${PIP_WHL_DIR}"/*.whl)

cp "${WHL_PATH}" "$(pwd)"/.
chmod +x tensorflow/tools/ci_build/builds/docker_cpu_pip.sh
docker run -e "BAZEL_VERSION=${BAZEL_VERSION}" -e "CI_BUILD_USER=$(id -u -n)" -e "CI_BUILD_UID=$(id -u)"  -e "CI_BUILD_GROUP=$(id -g -n)" -e "CI_BUILD_GID=$(id -g)"  -e "CI_BUILD_HOME=/bazel_pip" -v "$(pwd)":/bazel_pip tensorflow/tensorflow:devel-py3 "./bazel_pip/tensorflow/tools/ci_build/builds/with_the_same_user" "./bazel_pip/tensorflow/tools/ci_build/builds/docker_cpu_pip.sh"
