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

# Record GPU count and CUDA version status
if [[ "$TFCI_NVIDIA_SMI_ENABLE" == 1 ]]; then
  tfrun nvidia-smi
fi

# Update the version numbers for Nightly only
if [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]]; then
  python_bin=python3
  # TODO(belitskiy): Add a `python3` alias/symlink to Windows Docker image.
  if [[ $(uname -s) = MSYS_NT* ]]; then
    python_bin="python"
  fi
  tfrun "$python_bin" tensorflow/tools/ci_build/update_version.py --nightly
  # replace tensorflow to tf_nightly in the wheel name
  export TFCI_BUILD_PIP_PACKAGE_ARGS="$(echo $TFCI_BUILD_PIP_PACKAGE_ARGS | sed 's/tensorflow/tf_nightly/')"
fi

tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS build $TFCI_BAZEL_COMMON_ARGS --config=cuda_wheel //tensorflow/tools/pip_package:wheel $TFCI_BUILD_PIP_PACKAGE_ARGS

tfrun "$TFCI_FIND_BIN" ./bazel-bin/tensorflow/tools/pip_package -iname "*.whl" -exec cp {} $TFCI_OUTPUT_DIR \;
tfrun mkdir ./dist
tfrun cp $TFCI_OUTPUT_DIR/*.whl ./dist
tfrun bash ./ci/official/utilities/rename_and_verify_wheels.sh

if [[ "$TFCI_ARTIFACT_STAGING_GCS_ENABLE" == 1 ]]; then
  # Note: -n disables overwriting previously created files.
  # TODO(b/389744576): Remove when gsutil is made to work properly on MSYS2.
  if [[ $(uname -s) != MSYS_NT* ]]; then
    gsutil cp -n "$TFCI_OUTPUT_DIR"/*.whl "$TFCI_ARTIFACT_STAGING_GCS_URI"
  else
    powershell -command "gsutil cp -n '$TFCI_OUTPUT_DIR/*.whl' '$TFCI_ARTIFACT_STAGING_GCS_URI'"
  fi
fi

if [[ "$TFCI_WHL_BAZEL_TEST_ENABLE" == 1 ]]; then
  tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS test $TFCI_BAZEL_COMMON_ARGS $TFCI_BUILD_PIP_PACKAGE_ARGS --repo_env=TF_PYTHON_VERSION=$TFCI_PYTHON_VERSION --config="${TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX}_wheel_test"
fi
