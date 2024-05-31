#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
        ROCM_INSTALL_DIR=/opt/rocm-6.0.2
    else
        ROCM_INSTALL_DIR=$ROCM_PATH
    fi
fi

export PYTHON_BIN_PATH=`which python3`
export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR
TAGS_FILTER="gpu,requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-no_rocm"
UNSUPPORTED_GPU_TAGS="$(echo -requires-gpu-sm{60,70,80,86,89,90}{,-only})"
TAGS_FILTER="${TAGS_FILTER},${UNSUPPORTED_GPU_TAGS// /,}"

bazel \
    test \
    --config=rocm \
    --build_tag_filters=${TAGS_FILTER} \
    --test_tag_filters=${TAGS_FILTER} \
    --test_timeout=920,2400,7200,9600 \
    --test_sharding_strategy=disabled \
    --test_output=errors \
    --keep_going \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
    --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
    --action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
    --action_env=XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=true \
    --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute \
    -- //xla/...
