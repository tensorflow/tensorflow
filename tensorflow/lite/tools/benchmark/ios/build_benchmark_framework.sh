#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
BENCHMARK_DIR=tensorflow/lite/tools/benchmark
DEST_DIR="${BENCHMARK_DIR}/ios/TFLiteBenchmark/TFLiteBenchmark/Frameworks"
FRAMEWORK_TARGET=TensorFlowLiteBenchmarkC_framework

function usage() {
  echo "Usage: $(basename "$0") [-p]"
  echo "-p enable profiling"
  exit 1
}

PROFILING_ARGS=""
while getopts "p" opt_name; do
  case "$opt_name" in
    p) PROFILING_ARGS='--define=ruy_profiler=true';;
    *) usage;;
  esac
done
shift $(($OPTIND - 1))

function check_ios_configured() {
  if [ ! -f "${WORKSPACE_ROOT}/${BENCHMARK_DIR}/experimental/ios/BUILD" ]; then
    echo "ERROR: Benchmark framework BUILD file not found."
    echo "Please enable iOS support by running the \"./configure\" script" \
         "from the workspace root."
    exit 1
  fi
}

function build_framework() {
  set -x
  pushd "${WORKSPACE_ROOT}"

# Build the framework.
  bazel build --config=ios_fat -c opt ${PROFILING_ARGS} \
      "//${BENCHMARK_DIR}/experimental/ios:${FRAMEWORK_TARGET}"

# Copy the framework into the destination and unzip.
  mkdir -p "${DEST_DIR}"
  cp -f "bazel-bin/${BENCHMARK_DIR}/experimental/ios/${FRAMEWORK_TARGET}.zip" \
      "${DEST_DIR}"
  pushd "${DEST_DIR}"
  unzip -o "${FRAMEWORK_TARGET}.zip"
  rm -f "${FRAMEWORK_TARGET}.zip"

  popd
  popd
}

check_ios_configured
build_framework

