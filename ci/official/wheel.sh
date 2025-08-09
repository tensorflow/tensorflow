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
  export TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG="$(echo $TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG | sed 's/tensorflow/tf_nightly/')"
  export TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES="$(echo $TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES | sed 's/tensorflow/tf_nightly/g')"
fi

# TODO(b/361369076) Remove the following block after TF NumPy 1 is dropped
# Move hermetic requirement lock files for NumPy 1 to the root
if [[ "$TFCI_WHL_NUMPY_VERSION" == 1 ]]; then
  cp ./ci/official/requirements_updater/numpy1_requirements/*.txt .
fi

tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS build $TFCI_BAZEL_COMMON_ARGS --config=cuda_wheel //tensorflow/tools/pip_package:wheel $TFCI_BUILD_PIP_PACKAGE_BASE_ARGS $TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG --verbose_failures

tfrun "$TFCI_FIND_BIN" ./bazel-bin/tensorflow/tools/pip_package -iname "*.whl" -exec cp {} $TFCI_OUTPUT_DIR \;
tfrun mkdir -p ./dist
tfrun cp $TFCI_OUTPUT_DIR/*.whl ./dist
tfrun bash ./ci/official/utilities/rename_and_verify_wheels.sh

if [[ -n "$TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES" ]]; then
  # Re-build the wheel with the same config, but with different name(s), if any.
  # This is done after the rename_and_verify_wheel.sh run above, not to have
  # to contend with extra wheels there.
  for wheel_name in ${TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES}; do
    echo "Building for additional WHEEL_NAME: ${wheel_name}"
    CURRENT_WHEEL_NAME_ARG="--repo_env=WHEEL_NAME=${wheel_name}"
    tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS build $TFCI_BAZEL_COMMON_ARGS --config=cuda_wheel //tensorflow/tools/pip_package:wheel $TFCI_BUILD_PIP_PACKAGE_BASE_ARGS $CURRENT_WHEEL_NAME_ARG
    # Copy the wheel that was just created
    tfrun bash -c "$TFCI_FIND_BIN ./bazel-bin/tensorflow/tools/pip_package -iname "${wheel_name}*.whl" -printf '%T+ %p\n' | sort | tail -n 1 | awk '{print \$2}' | xargs -I {} cp {} $TFCI_OUTPUT_DIR"
  done
fi

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
  tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS test $TFCI_BAZEL_COMMON_ARGS $TFCI_BUILD_PIP_PACKAGE_BASE_ARGS $TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG --repo_env=TF_PYTHON_VERSION=$TFCI_PYTHON_VERSION --config="${TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX}_wheel_test"
fi
