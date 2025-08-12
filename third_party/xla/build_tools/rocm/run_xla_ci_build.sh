#!/usr/bin/env bash
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

CONFIG=$1
DISK_CACHE_PATH=$2

ASAN_ARGS=()
if [[ $CONFIG == "rocm_ci_hermetic" ]]; then
	ASAN_ARGS+=("--test_env=ASAN_OPTIONS=suppressions=$(realpath $(dirname $0))/asan_ignore_list.txt")
	ASAN_ARGS+=("--test_env=LSAN_OPTIONS=suppressions=$(realpath $(dirname $0))/lsan_ignore_list.txt")
	ASAN_ARGS+=("--config=asan")
fi

bazel --bazelrc=/usertools/rocm.bazelrc test \
	--config=${CONFIG} \
	--config=xla_cpp \
	--disk_cache=${DISK_CACHE_PATH} \
	--test_tag_filters=gpu,requires-gpu-amd,-requires-gpu-nvidia,-requires-gpu-intel,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-no_rocm,-requires-gpu-sm60,-requires-gpu-sm60-only,-requires-gpu-sm70,-requires-gpu-sm70-only,-requires-gpu-sm80,-requires-gpu-sm80-only,-requires-gpu-sm86,-requires-gpu-sm86-only,-requires-gpu-sm89,-requires-gpu-sm89-only,-requires-gpu-sm90,-requires-gpu-sm90-only \
	--build_tag_filters=gpu,requires-gpu-amd,-requires-gpu-nvidia,-requires-gpu-intel,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-no_rocm,-requires-gpu-sm60,-requires-gpu-sm60-only,-requires-gpu-sm70,-requires-gpu-sm70-only,-requires-gpu-sm80,-requires-gpu-sm80-only,-requires-gpu-sm86,-requires-gpu-sm86-only,-requires-gpu-sm89,-requires-gpu-sm89-only,-requires-gpu-sm90,-requires-gpu-sm90-only \
	--profile=/tf/pkg/profile.json.gz \
	--keep_going \
	--test_env=TF_TESTS_PER_GPU=1 \
	--test_env=TF_GPU_COUNT=2 \
	--action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
	--action_env=XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=true \
	--test_output=errors \
	--local_test_jobs=2 \
	--run_under=//tools/ci_build/gpu_build:parallel_gpu_execute \
	"${ASAN_ARGS[@]}"
