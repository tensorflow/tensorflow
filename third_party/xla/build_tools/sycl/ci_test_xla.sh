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
# ==============================================================================

# This script builds and executes tests. It can be run only on a system that
# has an Intel GPU with the appropriate driver installed.
# TEST_TARGETS="//xla/stream_executor/sycl/..." build_tools/sycl/ci_test_xla.sh

set -euo pipefail

no_of_gpus=$(ls /dev/dri/ | fgrep render | wc -l)
if [[ "${no_of_gpus}" -eq 0 ]]; then
  echo "unknown number of gpus."
  exit 1
fi

local_test_jobs=$((no_of_gpus * 8))

if [[ ${local_test_jobs} -gt 64 ]]; then
  local_test_jobs=64
  echo "local_test_jobs ${local_test_jobs} too high, using 64"
fi

TEST_TARGETS="${TEST_TARGETS:-\
  //xla/stream_executor/... \
  //xla/backends/gpu/codegen/emitters/tests/... \
  //xla/codegen/emitters/tests/... \
  -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_s4.hlo.test \
  -//xla/codegen/emitters/tests:loop/s8_to_s2.hlo.test \
  -//xla/backends/gpu/codegen/emitters/tests:transpose/multiple_roots_mixed_rank.hlo.test \
  -//xla/backends/gpu/codegen/emitters/tests:transpose/multiple_roots_one_shmem_transpose.hlo.test \
  -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_bf16.hlo.test \
  -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_two_heroes.hlo.test \
  -//xla/codegen/emitters/tests:loop/broadcast_constant.hlo.test\
}"

echo "TEST_TARGETS=${TEST_TARGETS}"

bazel test \
  --config=sycl_hermetic --verbose_failures -c opt \
  --test_timeout=900 --flaky_test_attempts=2 --keep_going --test_keep_going \
  --build_tag_filters=gpu,oneapi-only,requires-gpu-intel,-requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-cuda-only,-rocm-only,-no-oneapi \
  --test_tag_filters=gpu,oneapi-only,requires-gpu-intel,-requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-cuda-only,-rocm-only,-no-oneapi \
  -- \
  ${TEST_TARGETS}
