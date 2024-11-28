#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
# If rocm-smi exists locally (it should) use it to find
# out how many GPUs we have to test with.
rocm-smi -i
STATUS=$?
if [ $STATUS -ne 0 ]; then TF_GPU_COUNT=1; else
   TF_GPU_COUNT=$(rocm-smi -i|grep 'Device ID' |grep 'GPU' |wc -l)
fi
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
if [[ -n $1 ]]; then
    ROCM_INSTALL_DIR=$1
else
    if [[ -z "${ROCM_PATH}" ]]; then
        ROCM_INSTALL_DIR=/opt/rocm/
    else
        ROCM_INSTALL_DIR=$ROCM_PATH
    fi
fi

# Run configure.
export PYTHON_BIN_PATH=`which python3`

PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION
export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR

yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test \
      --config=rocm \
      -k \
      --test_tag_filters=gpu,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-cuda-only,-benchmark-test,-rocm_multi_gpu,-tpu,-v1only \
      --jobs=${N_BUILD_JOBS} \
      --local_test_jobs=${N_TEST_JOBS} \
      --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
      --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
      --test_env=MIOPEN_DEBUG_CONV_WINOGRAD=0 \
      --test_timeout 600,900,2400,7200 \
      --build_tests_only \
      --test_output=errors \
      --test_sharding_strategy=disabled \
      --test_size_filters=small,medium,large \
      --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
      -- \
      //tensorflow/... \
      -//tensorflow/core/tpu/... \
      -//tensorflow/lite/... \
      -//tensorflow/compiler/tf2tensorrt/... \
