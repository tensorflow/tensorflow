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

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh),gpu,-multi_gpu,-multi_gpu_h100,requires-gpu-amd,,-skip_rocprofiler_sdk,-no_oss,-oss_excluded,-oss_serial

if [ ! -d /tf/pkg ]; then
	mkdir -p /tf/pkg
fi

SCRIPT_DIR=$(dirname $0)
bazel --bazelrc="$SCRIPT_DIR/rocm_xla.bazelrc" test \
	"$@" \
	--build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
	--profile=/tf/pkg/profile.json.gz \
	--keep_going \
	--test_env=TF_TESTS_PER_GPU=1 \
	--action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
	--test_output=errors \
	--local_test_jobs=2 \
	--run_under=//build_tools/rocm:parallel_gpu_execute
