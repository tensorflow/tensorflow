#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# This script assumes the standard setup on tensorflow Jenkins windows machines.
# It is NOT guaranteed to work on any other machine. Use at your own risk!
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Anaconda3
# * Bazel windows executable copied as "bazel.exe" and included in PATH.

# All commands shall pass, and all should be visible.
set -x
set -e

# This script is under <repo_root>/tensorflow/tools/ci_build/windows/cpu/bazel
# Change into repository root.
script_dir=$(dirname $0)
cd ${script_dir%%tensorflow/tools/ci_build/windows/gpu/bazel}.

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# load bazel_test_lib.sh
source "tensorflow/tools/ci_build/windows/bazel/bazel_test_lib.sh" \
  || { echo "Failed to source bazel_test_lib.sh" >&2; exit 1; }

clean_output_base

run_configure_for_gpu_build

# Compiling the following test is extremely slow with -c opt
slow_compiling_test="//tensorflow/core/kernels:eigen_backward_spatial_convolutions_test"

# Find all the passing cc_tests on Windows and store them in a variable
passing_tests=$(bazel query "kind(cc_test, //tensorflow/cc/... + //tensorflow/core/...) - (${exclude_gpu_cc_tests}) - ($slow_compiling_test)" |
  # We need to strip \r so that the result could be store into a variable under MSYS
  tr '\r' ' ')

# TODO(pcloudy): There is a bug in Bazel preventing build with GPU support without -c opt
# Re-enable this test after it is fixed.
# bazel test --config=win-cuda -k $slow_compiling_test --test_output=errors
bazel test -c opt --config=win-cuda -k $passing_tests --test_output=errors
