#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

set -e

WORKSPACE_ROOT=$(bazel info workspace 2> /dev/null)
EVALUATION_DIR=tensorflow/lite/tools/evaluation/tasks
DEST_DIR="${EVALUATION_DIR}/ios/TFLiteEvaluation/TFLiteEvaluation/Frameworks"
FRAMEWORK_TARGET=TensorFlowLiteInferenceDiffC_framework

function usage() {
  echo "Usage: $(basename "$0")"
  exit 1
}

function check_ios_configured() {
  if [ ! -f "${WORKSPACE_ROOT}/${EVALUATION_DIR}/ios/BUILD" ]; then
    echo "ERROR: Inference Diff framework BUILD file not found."
    echo "Please enable iOS support by running the \"./configure\" script" \
         "from the workspace root."
    exit 1
  fi
}

function build_framework() {
  set -x
  pushd "${WORKSPACE_ROOT}"

# Build the framework.
  bazel build --config=ios_arm64 -c opt --cxxopt=-std=c++17 \
      "//${EVALUATION_DIR}/ios:${FRAMEWORK_TARGET}"

# Get the generated framework path.
BAZEL_OUTPUT_FILE_PATH=$(bazel cquery "//${EVALUATION_DIR}/ios:${FRAMEWORK_TARGET}" --config=ios_arm64 --output=starlark --starlark:expr="' '.join([f.path for f in target.files.to_list()])")

# Copy the framework into the destination and unzip.
  mkdir -p "${DEST_DIR}"
  cp -f "${BAZEL_OUTPUT_FILE_PATH}" \
      "${DEST_DIR}"
  pushd "${DEST_DIR}"
  unzip -o "${FRAMEWORK_TARGET}.zip"
  rm -f "${FRAMEWORK_TARGET}.zip"

  popd
  popd
}

check_ios_configured
build_framework
