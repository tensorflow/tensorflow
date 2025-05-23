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
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.
# set -o pipefail # Causes pipelines to fail if any command fails (see Run script)

echo "--- Configuring and Building Binaries ---"
echo "Building binaries for $HARDWARE_CATEGORY..."

# --- Configure ---
echo "Configuring backend..."
if [[ "$HARDWARE_CATEGORY" == CPU* ]]; then
  ./configure.py --backend=CPU || echo "INFO: CPU Configure script failed or is not applicable."
elif [[ "$HARDWARE_CATEGORY" == GPU* ]]; then
   ./configure.py --backend=CUDA --cuda_compiler=nvcc || echo "INFO: GPU Configure script failed or is not applicable."
else
  echo "::error::Unsupported hardware category for configuration: $HARDWARE_CATEGORY"
  exit 1
fi
echo "Configuration step finished."

# --- Determine Paths and Build ---
declare BAZEL_BIN_DIR="bazel-bin"
declare runner_binary_path=""
declare stats_binary_path=""
declare device_type_flag_value=""
declare bazel_exit_code=0

# TODO(juliagmt): use build.py to build binaries.
if [[ "$HARDWARE_CATEGORY" == CPU* ]]; then
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main"
    device_type_flag_value="host"

    echo "Building CPU binaries with RBE..."
     bazel build \
        --build_tag_filters=-no_oss,-gpu,-requires-gpu-nvidia,-requires-gpu-amd \
        --test_tag_filters=-no_oss,-gpu,-requires-gpu-nvidia,-requires-gpu-amd \
        --config=warnings \
        --config=nonccl \
        --config=rbe_linux_cpu \
        --color=yes \
        --test_output=errors \
        --verbose_failures \
        --keep_going \
        --nobuild_tests_only \
        --profile=profile.json.gz \
        --flaky_test_attempts=3 \
        --jobs=150 \
        --bes_upload_mode=fully_async \
        //xla/tools/multihost_hlo_runner:hlo_runner_main \
        //xla/tools:compute_xspace_stats_main
      bazel_exit_code=$?

elif [[ "$HARDWARE_CATEGORY" == GPU* ]]; then
    runner_binary_path="./$BAZEL_BIN_DIR/xla/tools/multihost_hlo_runner/hlo_runner_main_gpu"
    stats_binary_path="./$BAZEL_BIN_DIR/xla/tools/compute_xspace_stats_main_gpu"
    device_type_flag_value="gpu"

    echo "Building GPU binaries with RBE..."
     bazel build \
        --build_tag_filters=-no_oss,requires-gpu-nvidia,gpu,-rocm-only \
        --test_tag_filters=-no_oss,requires-gpu-nvidia,gpu,-rocm-only,requires-gpu-sm75-only,requires-gpu-sm60,requires-gpu-sm70,-requires-gpu-sm80,-requires-gpu-sm80-only,-requires-gpu-sm90,-requires-gpu-sm90-only,-requires-gpu-sm100,-requires-gpu-sm100-only,-requires-gpu-amd \
        --config=warnings --config=rbe_linux_cuda_nvcc \
        --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=7.5 \
        --run_under=//build_tools/ci:parallel_gpu_execute \
        --@cuda_driver//:enable_forward_compatibility=false --color=yes \
        --test_output=errors --verbose_failures --keep_going --nobuild_tests_only \
        --profile=profile.json.gz --flaky_test_attempts=3 --jobs=150 \
        --bes_upload_mode=fully_async \
         -- //xla/tools/multihost_hlo_runner:hlo_runner_main_gpu //xla/tools:compute_xspace_stats_main_gpu
      bazel_exit_code=$?
else
    echo "::error::Unsupported hardware category for building binaries: $HARDWARE_CATEGORY"
    exit 1
fi
 # Check build result
 if [ $bazel_exit_code -ne 0 ]; then 
   echo "::error::Bazel build failed with exit code $bazel_exit_code!"
   exit $bazel_exit_code
 fi
 echo "Bazel build completed successfully."

# --- Verify and Output ---
echo "Verifying binary existence..."
if [ ! -f "$runner_binary_path" ]; then echo "::error::Runner binary '$runner_binary_path' not found after build!"; exit 1; fi
if [ ! -f "$stats_binary_path" ]; then echo "::error::Stats binary '$stats_binary_path' not found after build!"; exit 1; fi
echo "Binaries verified."
 
echo "Setting step outputs..."
echo "runner_binary=$runner_binary_path" >> "$GITHUB_OUTPUT"
echo "stats_binary=$stats_binary_path" >> "$GITHUB_OUTPUT"
echo "device_type_flag=$device_type_flag_value" >> "$GITHUB_OUTPUT"

echo "  runner_binary=$runner_binary_path"
echo "  stats_binary=$stats_binary_path"
echo "  device_type_flag=$device_type_flag_value"
echo "--- Build Script Finished ---"
