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

install_ubuntu_16_pip_deps pip2.7
# Update bazel
update_bazel_linux

# Run configure.
export TF_NEED_GCP=1
export TF_NEED_HDFS=1
export TF_NEED_S3=1
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=10.1
export TF_CUDNN_VERSION=7
export TF_NEED_TENSORRT=1
export TENSORRT_INSTALL_PATH=/usr/local/tensorrt
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=$(which python2.7)
export TF2_BEHAVIOR=1
export PROJECT_NAME="tensorflow_gpu"
export LD_LIBRARY_PATH="/usr/local/cuda:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$TENSORRT_INSTALL_PATH/lib"
export TF_CUDA_COMPUTE_CAPABILITIES=3.5,3.7,5.2,6.0,6.1,7.0

yes "" | "$PYTHON_BIN_PATH" configure.py

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/PRESUBMIT_BUILD_TARGETS.sh

tag_filters="gpu,requires-gpu,-no_gpu,-nogpu,-no_oss,-oss_serial,-no_oss_py2"

bazel test --config=cuda --config=opt \
  --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.1:toolchain \
  --linkopt=-lrt \
  --action_env=TF2_BEHAVIOR="${TF2_BEHAVIOR}" \
  --test_lang_filters=py \
  --build_tag_filters="${tag_filters}" \
  --test_tag_filters="${tag_filters}" \
  --test_timeout="300,450,1200,3600" --local_test_jobs=4 \
  --test_output=errors --verbose_failures=true --keep_going \
  --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
  -- ${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/...
