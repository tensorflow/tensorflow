#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# Extract hermetic CUDA User-Mode Driver (UMD) flags
HERMETIC_CUDA_UMD_FLAGS=""
if [[ "$TFCI_BAZEL_HERMETIC_CUDA_UMD_ENABLE" == 1 ]]; then
  HERMETIC_CUDA_UMD_FLAGS="--@local_config_cuda//cuda:override_include_cuda_libs=true --config=hermetic_cuda_umd"
fi

PROFILE_JSON_PATH="$TFCI_OUTPUT_DIR/profile.json.gz"
if [[ $(uname -s) == "MSYS_NT"* ]] || [[ $(uname -s) == "MINGW64_NT"* ]]; then
  if [[ -n "$TFCI_OUTPUT_WIN_DOCKER_DIR" ]]; then
    PROFILE_JSON_PATH=$(replace_drive_letter_with_prefix "$TFCI_OUTPUT_WIN_DOCKER_DIR")
    PROFILE_JSON_PATH="$PROFILE_JSON_PATH/profile.json.gz"
  elif [[ "$TFCI_GITHUB_ACTIONS" == "true" ]]; then
    PROFILE_JSON_PATH=$(cygpath -w "$PROFILE_JSON_PATH")
  fi
fi

# TODO(b/361369076) Remove the following block after TF NumPy 1 is dropped
# Move hermetic requirement lock files for NumPy 1 to the root
if [[ "$TFCI_WHL_NUMPY_VERSION" == 1 ]]; then
  cp ./ci/official/requirements_updater/numpy1_requirements/*.txt .
fi

if [[ $TFCI_PYCPP_SWAP_TO_BUILD_ENABLE == 1 ]]; then
  tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS build $TFCI_BAZEL_COMMON_ARGS --profile "$PROFILE_JSON_PATH" $HERMETIC_CUDA_UMD_FLAGS --config="${TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX}_pycpp_test"
else
  tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS test $TFCI_BAZEL_COMMON_ARGS --profile "$PROFILE_JSON_PATH" $HERMETIC_CUDA_UMD_FLAGS --config="${TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX}_pycpp_test"
fi

# Note: the profile can be viewed by visiting chrome://tracing in a Chrome browser.
# See https://docs.bazel.build/versions/main/skylark/performance.html#performance-profiling
tfrun bazel analyze-profile "$PROFILE_JSON_PATH"
