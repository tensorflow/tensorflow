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
source tensorflow/tools/ci_build/ctpu/ctpu.sh

install_ubuntu_16_python_pip_deps python3.7
install_bazelisk
install_ctpu pip3.7

# Run configure.
export TF_NEED_GCP=1
export TF_NEED_HDFS=1
export TF_NEED_S3=1
export TF_NEED_CUDA=0
export TF_NEED_TENSORRT=0
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=$(which python3.7)
export TF2_BEHAVIOR=1

yes "" | "$PYTHON_BIN_PATH" configure.py

test_patterns=(//tensorflow/... -//tensorflow/compiler/... -//tensorflow/lite/...)
tag_filters="tpu,-tpu_pod,-no_tpu,-notpu,-no_oss,-no_oss_py37"

bazel_args=(
  --config=opt \
  --config=short_logs \
  --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.1:toolchain \
  --linkopt=-lrt \
  --action_env=TF2_BEHAVIOR="${TF2_BEHAVIOR}" \
  --noincompatible_strict_action_env \
  --build_tag_filters="${tag_filters}" \
  --test_tag_filters="${tag_filters}" \
  --test_output=errors --verbose_failures=true --keep_going
)

bazel build "${bazel_args[@]}" -- "${test_patterns[@]}"

ctpu_up -s v2-8 -p tensorflow-testing-tpu

test_args=(
  --test_timeout=120,600,-1,-1 \
  --test_arg=--tpu="${TPU_NAME}" \
  --test_arg=--zone="${TPU_ZONE}" \
  --test_arg=--test_dir_base=gs://kokoro-tpu-testing/tempdir/ \
  --local_test_jobs=1
)

set +e
bazel test "${bazel_args[@]}" "${test_args[@]}" -- "${test_patterns[@]}"
test_xml_summary_exit
