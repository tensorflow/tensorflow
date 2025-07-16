# Copyright 2025 The OpenXLA Authors. All Rights Reserved.
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
# ============================================================================
# .github/workflows/benchmarks/build_binaries.sh
# TODO(juliagmt): convert this to a python script.
# Builds HLO runner and stats computation binaries using build.py.
# Expects HARDWARE_CATEGORY (matching enum values like CPU_X86, GPU_L4, etc.)
# and OUTPUT_DIR environment variables to be set.
# Outputs: runner_binary, stats_binary, device_type_flag to GITHUB_OUTPUT
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.
# set -o pipefail # Causes pipelines to fail if any command fails (see Run script)

echo "--- Configuring and Building Binaries ---"
echo "Workspace: $(pwd)"
echo "Hardware Category: $HARDWARE_CATEGORY"
echo "Output Directory (for profile.json.gz): $OUTPUT_DIR"

# Sanitize HARDWARE_CATEGORY for use in filenames.
# Replaces non-alphanumeric with underscore, converts to lowercase.
HW_CATEGORY_SLUG=$(echo "${HARDWARE_CATEGORY:-UNSPECIFIED}" | tr '[:upper:]' '[:lower:]' | sed 's/[^A-Z0-9_]/_/g')

# --- 1. Configure Backend (using ./configure.py if available) ---
# This part can remain, as build.py doesn't currently handle `configure.py`
configure_backend() {
  echo "Configuring backend using ./configure.py if present..."
  if [ ! -f "./configure.py" ]; then
    echo "INFO: ./configure.py not found. Skipping configuration step."
    return
  fi

  local hw_category_upper_for_configure
  hw_category_upper_for_configure=$(echo "${HARDWARE_CATEGORY:-UNSPECIFIED}" | tr '[:lower:]' '[:upper:]')

  case "$hw_category_upper_for_configure" in
    CPU_X86 | CPU_ARM64)
      echo "Running: ./configure.py --backend=CPU"
      ./configure.py --backend=CPU || echo "INFO: CPU Configure script failed or is not applicable."
      ;;
    GPU_L4 | GPU_B200)
      echo "Running: ./configure.py --backend=CUDA --cuda_compiler=nvcc"
      ./configure.py --backend=CUDA --cuda_compiler=nvcc || echo "INFO: GPU Configure script failed or is not applicable."
      ;;
    *)
      echo "INFO: Unknown hardware category '$hw_category_upper_for_configure'"
      ;;
  esac
  echo "Configuration step finished."
}

# --- 2. Main Build Logic using build.py ---
declare BAZEL_BIN_DIR="bazel-bin"
declare runner_binary_path=""
declare stats_binary_path=""
declare device_type_flag_value=""

configure_backend

echo "Building with build.py for HARDWARE_CATEGORY: $HARDWARE_CATEGORY"

BUILD_TYPE=""
case "$HARDWARE_CATEGORY" in
  CPU_X86)
    BUILD_TYPE="XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS"
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main"
    device_type_flag_value="host"
    ;;
  CPU_ARM64)
    BUILD_TYPE="XLA_LINUX_ARM64_CPU_48_VCPU_PRESUBMIT_GITHUB_ACTIONS"
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main"
    device_type_flag_value="host"
    ;;
  GPU_L4)
    BUILD_TYPE="XLA_LINUX_X86_GPU_L4_16_VCPU_PRESUBMIT_GITHUB_ACTIONS" # Or _48_VCPU if that's the more common
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main_gpu"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main_gpu"
    device_type_flag_value="gpu"
    ;;
  GPU_B200)
    BUILD_TYPE="XLA_LINUX_X86_GPU_A4_224_VCPU_PRESUBMIT_GITHUB_ACTIONS"
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main_gpu"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main_gpu"
    device_type_flag_value="gpu"
    ;;
  *)
    echo "::error::Unsupported HARDWARE_CATEGORY: '$HARDWARE_CATEGORY'. This script is configured to handle specific values from the HardwareCategory enum (CPU_X86, CPU_ARM64, GPU_L4, GPU_B200)."
    exit 1
    ;;
esac

echo "Executing build with build.py for build type: $BUILD_TYPE"
# Run build.py with the determined build type
python3 build_tools/ci/build.py --build="$BUILD_TYPE" || {
  echo "::error::build.py failed for $BUILD_TYPE!"
  exit 1
}


# --- 3. Verify Binaries and Set Outputs ---
echo "Verifying binary existence..."
if [ -z "$runner_binary_path" ] || [ ! -f "$runner_binary_path" ]; then
  echo "::error::Runner binary path not set or binary '$runner_binary_path' not found after build for $HARDWARE_CATEGORY!"
  exit 1
fi
if [ -z "$stats_binary_path" ] || [ ! -f "$stats_binary_path" ]; then
  echo "::error::Stats binary path not set or binary '$stats_binary_path' not found after build for $HARDWARE_CATEGORY!"
  exit 1
fi
echo "Binaries verified: $runner_binary_path, $stats_binary_path"

echo "Setting step outputs..."
{
  echo "runner_binary=$runner_binary_path"
  echo "stats_binary=$stats_binary_path"
  echo "device_type_flag=$device_type_flag_value"
} >> "$GITHUB_OUTPUT"

echo "  Runner binary: $runner_binary_path"
echo "  Stats binary: $stats_binary_path"
echo "  Device type flag: $device_type_flag_value"
echo "--- Build Script Finished ---"