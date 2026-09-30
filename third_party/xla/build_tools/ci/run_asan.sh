#!/usr/bin/env bash
# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# Runs XLA CPU AddressSanitizer (ASAN) tests with Python 3.14.
#
# Usage:
#   # Run locally with the fast curated suite (no RBE required):
#   ./build_tools/ci/run_asan.sh
#
#   # Run specific targets locally:
#   ./build_tools/ci/run_asan.sh //xla/python/transfer/... //xla/python_api/...
#
#   # Run the full XLA CPU suite on RBE (default in GitHub Actions CI):
#   ./build_tools/ci/run_asan.sh --rbe --suite=full
#
#   # Run the full XLA CPU suite locally without RBE:
#   ./build_tools/ci/run_asan.sh --no-rbe --suite=full

set -euo pipefail
set -x

PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.14}"
USE_RBE="${USE_RBE:-}"
SUITE="${SUITE:-}"
CUSTOM_TARGETS=()
BAZEL_EXTRA_FLAGS=()

if [[ -n "${EXTRA_BAZEL_ARGS:-}" ]]; then
  read -r -a BAZEL_EXTRA_FLAGS <<< "${EXTRA_BAZEL_ARGS}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rbe)
      USE_RBE="true"
      shift
      ;;
    --rbe=*)
      USE_RBE="${1#*=}"
      shift
      ;;
    --no-rbe)
      USE_RBE="false"
      shift
      ;;
    --suite=*)
      SUITE="${1#*=}"
      shift
      ;;
    --python-version=*)
      PYTHON_VERSION="${1#*=}"
      shift
      ;;
    --bazel-arg=*)
      BAZEL_EXTRA_FLAGS+=("${1#*=}")
      shift
      ;;
    --help|-h)
      sed -n '16,29p' "$0"
      exit 0
      ;;
    --)
      shift
      CUSTOM_TARGETS+=("$@")
      break
      ;;
    *)
      CUSTOM_TARGETS+=("$1")
      shift
      ;;
  esac
done

# Default USE_RBE to true in GitHub Actions (where the linux-x86-n2-* runner
# pool has GCE default credentials for projects/tensorflow-testing), and false
# for local developer runs.
if [[ -z "${USE_RBE}" ]]; then
  if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
    USE_RBE="true"
  else
    USE_RBE="false"
  fi
fi

# Default SUITE to "full" when RBE or RBE cache is enabled, and "quick" when
# running locally.
if [[ -z "${SUITE}" ]]; then
  if [[ "${USE_RBE}" != "false" ]]; then
    SUITE="full"
  else
    SUITE="quick"
  fi
fi

FULL_SUITE_TARGETS=(
  "//xla/..."
  "//build_tools/..."
  "-//xla/backends/gpu/..."
)

# Fast, high-signal ASAN suite covering core XLA data structures,
# Python/C++ pybind extensions, PJRT/transfer concurrency, and TSL primitives
# without pulling in heavy MLIR/SPIR-V/StableHLO dialect codegen.
QUICK_SUITE_TARGETS=(
  "//xla:status_macros_test"
  "//xla:shape_util_test"
  "//xla:literal_test"
  "//xla/python_api:xla_shape_test"
  "//xla/python_api:xla_literal_test"
  "//xla/python/transfer:event_loop_test"
  "//xla/python/transfer:socket_bulk_transport_test"
  "//xla/python/transfer:socket-server_test"
  "//xla/python/transfer:streaming_test"
  "//xla/tsl/concurrency/..."
  "//xla/tsl/lib/gtl/..."
  "//xla/backends/cpu/runtime:kernel_test"
)

if [[ ${#CUSTOM_TARGETS[@]} -gt 0 ]]; then
  TARGETS=("${CUSTOM_TARGETS[@]}")
elif [[ -n "${TARGETS:-}" ]]; then
  # Allow space-separated TARGETS environment variable override.
  read -r -a TARGETS <<< "${TARGETS}"
elif [[ "${SUITE}" == "full" ]]; then
  TARGETS=("${FULL_SUITE_TARGETS[@]}")
elif [[ "${SUITE}" == "quick" ]]; then
  TARGETS=("${QUICK_SUITE_TARGETS[@]}")
else
  echo "Unknown --suite='${SUITE}' (expected 'full' or 'quick')" >&2
  exit 1
fi

EXEC_FLAGS=()
if [[ "${USE_RBE}" == "true" ]]; then
  EXEC_FLAGS=(
    "--config=rbe_linux_cpu"
    "--remote_download_minimal"
    "--jobs=150"
  )
elif [[ "${USE_RBE}" == "compile_only" ]]; then
  EXEC_FLAGS=(
    "--config=rbe_linux_cpu"
    "--strategy=TestRunner=local"
    "--local_test_jobs=HOST_CPUS"
    "--local_resources=memory=HOST_RAM*.8"
    "--local_resources=cpu=HOST_CPUS"
  )
elif [[ "${USE_RBE}" == "cache" ]]; then
  EXEC_FLAGS=(
    "--config=resultstore"
    "--remote_cache=grpcs://remotebuildexecution.googleapis.com"
    "--remote_instance_name=projects/tensorflow-testing/instances/default_instance"
    "--remote_upload_local_results=true"
    "--local_resources=memory=HOST_RAM*.8"
    "--local_resources=cpu=HOST_CPUS"
  )
elif [[ "${USE_RBE}" == "false" ]]; then
  EXEC_FLAGS=(
    "--local_resources=memory=HOST_RAM*.8"
    "--local_resources=cpu=HOST_CPUS"
  )
else
  echo "Unknown USE_RBE='${USE_RBE}' (expected 'true', 'compile_only', 'cache', or 'false')" >&2
  exit 1
fi

echo "=== XLA ASAN Test Runner ==="
echo "  Python version : ${PYTHON_VERSION}"
echo "  Use RBE        : ${USE_RBE}"
echo "  Suite          : ${SUITE}"
echo "  Targets        : ${TARGETS[*]}"
if [[ ${#BAZEL_EXTRA_FLAGS[@]} -gt 0 ]]; then
  echo "  Extra flags    : ${BAZEL_EXTRA_FLAGS[*]}"
fi

exec bazel test \
  --repo_env="HERMETIC_PYTHON_VERSION=${PYTHON_VERSION}" \
  --config=asan_shared \
  --config=nonccl \
  --run_under=//build_tools/ci:asan_wrapper \
  --color=yes \
  --test_output=errors \
  --verbose_failures \
  --keep_going \
  --build_tests_only \
  --test_timeout=1800 \
  --show_progress_rate_limit=10 \
  --profile=profile.json.gz \
  --build_tag_filters=-no_oss,-gpu,-requires-gpu-nvidia,-requires-gpu-amd,-requires-gpu-intel,-noasan,-nosan \
  --test_tag_filters=-no_oss,-gpu,-requires-gpu-nvidia,-requires-gpu-amd,-requires-gpu-intel,-noasan,-nosan \
  --//xla/tsl:ci_build=true \
  "${EXEC_FLAGS[@]}" \
  "${BAZEL_EXTRA_FLAGS[@]}" \
  -- "${TARGETS[@]}"
