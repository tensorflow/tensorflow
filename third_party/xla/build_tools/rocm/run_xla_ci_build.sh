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

SCRIPT_DIR=$(dirname $0)
bazel --bazelrc="$SCRIPT_DIR/rocm_xla.bazelrc" test \
	"$@" \
	--test_tag_filters=gpu,requires-gpu-amd,-requires-gpu-nvidia,-requires-gpu-intel,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-no_rocm,-requires-gpu-sm60,-requires-gpu-sm60-only,-requires-gpu-sm70,-requires-gpu-sm70-only,-requires-gpu-sm80,-requires-gpu-sm80-only,-requires-gpu-sm86,-requires-gpu-sm86-only,-requires-gpu-sm89,-requires-gpu-sm89-only,-requires-gpu-sm90,-requires-gpu-sm90-only \
	--build_tag_filters=gpu,requires-gpu-amd,-requires-gpu-nvidia,-requires-gpu-intel,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-no_rocm,-requires-gpu-sm60,-requires-gpu-sm60-only,-requires-gpu-sm70,-requires-gpu-sm70-only,-requires-gpu-sm80,-requires-gpu-sm80-only,-requires-gpu-sm86,-requires-gpu-sm86-only,-requires-gpu-sm89,-requires-gpu-sm89-only,-requires-gpu-sm90,-requires-gpu-sm90-only \
	--profile=/tf/pkg/profile.json.gz \
	--keep_going \
	--test_env=TF_TESTS_PER_GPU=1 \
	--action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
	--action_env=XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=true \
	--test_output=errors \
	--local_test_jobs=2 \
	--run_under=//build_tools/rocm:parallel_gpu_execute
