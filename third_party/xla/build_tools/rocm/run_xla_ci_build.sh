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
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh)

mkdir -p /tf/pkg

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,-multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_rocm_cpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-requires-gpu-rocm,-requires-gpu-amd"
    fi
    if [[ "$arg" == "--config=no_flaky" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-flaky"
    fi
done

SCRIPT_DIR=$(dirname $0)
BAZEL_ARGS=()
if [[ -n "${XLA_ROCM_TARGET_PATTERN_FILE:-}" && -f "${XLA_ROCM_TARGET_PATTERN_FILE}" ]]; then
    TAG_FILTERS="${TAG_FILTERS},-manual"
    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            break
        fi
        BAZEL_ARGS+=("$arg")
    done
    BAZEL_ARGS+=(
        "--skip_incompatible_explicit_targets"
        "--build_tests_only"
        "--target_pattern_file=${XLA_ROCM_TARGET_PATTERN_FILE}"
    )
else
    BAZEL_ARGS=("$@")
fi

set +e
bazel --bazelrc="$SCRIPT_DIR/rocm_xla_ci.bazelrc" test \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --nokeep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16 --xla_gpu_blas_max_algorithms=8" \
    --test_output=errors \
    --sandbox_add_mount_pair=/dev/null:/etc/ld.so.cache \
    --curses=no \
    --color=yes \
    "${BAZEL_ARGS[@]}"
BAZEL_RC=$?
set -e

if [[ $BAZEL_RC -eq 4 && -n "${XLA_ROCM_TARGET_PATTERN_FILE:-}" ]]; then
    echo "Bazel returned exit code 4 (no matching test targets in ${XLA_ROCM_TARGET_PATTERN_FILE}); treating as success."
    exit 0
fi
exit $BAZEL_RC
