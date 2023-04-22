#!/usr/bin/env bash
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
#
# ==============================================================================

set -e
set -x

N_JOBS=$(grep -c ^processor /proc/cpuinfo)
LT_JOBS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo "Bazel will use ${LT_JOBS} local test job(s)."
echo ""

# Run configure.
export PYTHON_BIN_PATH=`which python3`
export CC_OPT_FLAGS='-mcpu=power8 -mtune=power8'

export TF_NEED_CUDA=1
export TF_CUDA_COMPUTE_CAPABILITIES=3.7

yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test --config=cuda --test_tag_filters=-no_oss,-oss_serial,-no_gpu,-benchmark-test -k \
    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 \
    --test_output=errors --local_test_jobs=${LT_JOBS} --build_tests_only --config=opt \
    --test_size_filters=small,medium \
    --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute -- \
    //tensorflow/... -//tensorflow/compiler/...
