#!/bin/bash
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
#
# ==============================================================================
#
# Run the python unit tests from the source code on the pip installation.
#
# Usage:
#   run_pip_tests.sh [--virtualenv] [--gpu] [--mac]
#
# If the flag --virtualenv is set, the script will use "python" as the Python
# binary path. Otherwise, it will use tools/python_bin_path.sh to determine
# the Python binary path.
#
# The --gpu flag informs the script that this is a GPU build, so that the
# appropriate test blacklists can be applied accordingly.
#
# The --mac flag informs the script that this is running on mac. Mac does not
# have flock, so we should skip using parallel_gpu_execute on mac.
#
#   TF_BUILD_APPEND_ARGUMENTS:
#                      Additional command line arguments for the bazel,
#                      pip.sh or android.sh command

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"

# Process input arguments
IS_VIRTUALENV=0
IS_GPU=0
IS_MAC=0
while true; do
  if [[ "$1" == "--virtualenv" ]]; then
    IS_VIRTUALENV=1
  elif [[ "$1" == "--gpu" ]]; then
    IS_GPU=1
  elif [[ "$1" == "--mac" ]]; then
    IS_MAC=1
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

TF_GPU_COUNT=${TF_GPU_COUNT:-8}

# PIP tests should have a "different" path. Different than the one we place
# virtualenv, because we are deleting and recreating it here.
PIP_TEST_PREFIX=bazel_pip
PIP_TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
rm -rf $PIP_TEST_ROOT
mkdir -p $PIP_TEST_ROOT
ln -s $(pwd)/tensorflow ${PIP_TEST_ROOT}/tensorflow

# Do not run tests with "no_pip" tag. If running GPU tests, also do not run
# tests with no_pip_gpu tag.
PIP_TEST_FILTER_TAG="-no_pip"
if [[ ${IS_GPU} == "1" ]]; then
  PIP_TEST_FILTER_TAG="-no_pip_gpu,${PIP_TEST_FILTER_TAG}"
fi

# Bazel flags we need for all tests:
#     define=no_tensorflow_py_deps=true, to skip all test dependencies.
#     test_lang_filters=py only py tests for pip package testing
#     TF_BUILD_APPEND_ARGUMENTS any user supplied args.
BAZEL_FLAGS="--define=no_tensorflow_py_deps=true --test_lang_filters=py \
  --build_tests_only -k --test_tag_filters=${PIP_TEST_FILTER_TAG} \
  --test_timeout 300,450,1200,3600 ${TF_BUILD_APPEND_ARGUMENTS}"

BAZEL_TEST_TARGETS="//${PIP_TEST_PREFIX}/tensorflow/contrib/... \
  //${PIP_TEST_PREFIX}/tensorflow/python/... \
  //${PIP_TEST_PREFIX}/tensorflow/tensorboard/..."

# Run configure again, we might be using a different python path, due to
# virtualenv.
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_ENABLE_XLA=${TF_BUILD_ENABLE_XLA:-0}

# Obtain the path to Python binary
if [[ ${IS_VIRTUALENV} == "1" ]]; then
  PYTHON_BIN_PATH="$(which python)"
else
  source tools/python_bin_path.sh
  # Assume: PYTHON_BIN_PATH is exported by the script above
fi

export TF_NEED_CUDA=$IS_GPU
yes "" | ./configure

# Figure out how many concurrent tests we can run and do run the tests.
BAZEL_PARALLEL_TEST_FLAGS=""
if [[ $IS_GPU == 1 ]]; then
  # Number of test threads is the number of GPU cards available.
  if [[ $IS_MAC == 1 ]]; then
    BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=1"
  else
    PAR_TEST_JOBS=$TF_GPU_COUNT
    BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=${TF_GPU_COUNT} \
        --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute"
  fi
else
  # Number of test threads is the number of physical CPUs.
  if [[ $IS_MAC == 1 ]]; then
    BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=$(sysctl -n hw.ncpu)"
  else
    BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=$(grep -c ^processor /proc/cpuinfo)"
  fi
fi

# Actually run the tests.
bazel test ${BAZEL_FLAGS} ${BAZEL_PARALLEL_TEST_FLAGS} -- \
    ${BAZEL_TEST_TARGETS}

