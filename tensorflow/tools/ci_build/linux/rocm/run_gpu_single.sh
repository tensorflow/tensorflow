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

SCRIPT_DIR=$(realpath $(dirname $0))
# Get base ROCm tag filters (all the exclusions)
TAG_FILTERS=$(${SCRIPT_DIR}/rocm_tag_filters.sh)
# Add TensorFlow-specific filters
# Note: requires-gpu-rocm is added as positive filter (tf_gpu_tests_tags() adds this during ROCm builds)
#       rocm_multi_gpu is excluded as we're running single GPU tests
TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,-rocm_multi_gpu"

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test \
      --config=rocm_ci_hermetic \
      -k \
      --build_tag_filters=${TAG_FILTERS} \
      --test_tag_filters=${TAG_FILTERS} \
      --jobs=200 \
      --local_test_jobs=4 \
      --repo_env=ROCM_PATH="" \
      --test_env=TF_GPU_COUNT=${TF_GPU_COUNT} \
      --test_env=TF_TESTS_PER_GPU=1 \
      --test_timeout 600,900,2400,7200 \
      --build_tests_only \
      --test_output=errors \
      --test_sharding_strategy=disabled \
      --test_size_filters=small,medium,large \
      --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
      "$@" \
      -- \
      //tensorflow/... \
      -//tensorflow/core/tpu/... \
      -//tensorflow/lite/... \
      -//tensorflow/compiler/tf2tensorrt/... \
