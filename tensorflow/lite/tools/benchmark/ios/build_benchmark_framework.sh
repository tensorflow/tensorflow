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
set -x

WORKSPACE_ROOT=$(bazel info workspace)
BENCHMARK_DIR=tensorflow/lite/tools/benchmark
DEST_DIR="${BENCHMARK_DIR}/ios/TFLiteBenchmark/TFLiteBenchmark/Frameworks"
FRAMEWORK_TARGET=TensorFlowLiteBenchmarkC_framework

usage() {
  echo "Usage: $(basename "$0") [-p]"
  echo "-p enable profiling"
  exit 1
}

PROFILING_ARGS=""
while getopts "p" opt_name; do
  case "$opt_name" in
    p) PROFILING_ARGS='--copt=-DGEMMLOWP_PROFILING';;
    *) usage;;
  esac
done
shift $(($OPTIND - 1))

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
