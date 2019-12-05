#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

N_JOBS=$(grep -c ^processor /proc/cpuinfo)
N_GPUS=$(lspci|grep 'controller'|grep 'AMD/ATI'|wc -l)

echo ""
echo "Bazel will use ${N_JOBS} concurrent build job(s) and ${N_GPUS} concurrent test job(s)."
echo ""

# Run configure.
export PYTHON_BIN_PATH=`which python3`
export CC_OPT_FLAGS='-mavx'

export TF_NEED_ROCM=1
export TF_GPU_COUNT=${N_GPUS}

yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test \
      --config=rocm \
      -k \
      --test_tag_filters=gpu,-no_gpu,-no_rocm,-benchmark-test,-no_oss,-oss_serial,-rocm_multi_gpu, \
      --test_timeout 600,900,2400,7200 \
      --test_output=errors \
      --jobs=${N_JOBS} \
      --local_test_jobs=${TF_GPU_COUNT} \
      --test_sharding_strategy=disabled \
      --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
      -- \
      //tensorflow/... \
      -//tensorflow/compiler/... \
      -//tensorflow/lite/... \
      -//tensorflow/python/compiler/tensorrt/... \
&& bazel test \
      --config=rocm \
      -k \
      --test_tag_filters=gpu \
      --test_timeout 600,900,2400,7200 \
      --test_output=errors \
      --jobs=${N_JOBS} \
      --local_test_jobs=1 \
      --test_sharding_strategy=disabled \
      -- \
      //tensorflow/core/nccl:nccl_manager_test

