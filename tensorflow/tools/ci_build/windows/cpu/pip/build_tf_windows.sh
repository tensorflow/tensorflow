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

# This script is under <repo_root>/tensorflow/tools/ci_build/windows/cpu/pip/
# Change into repository root.
script_dir=$(dirname $0)
cd ${script_dir%%tensorflow/tools/ci_build/windows/cpu/pip}.

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/cpu/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# bazel clean --expunge doesn't work on Windows yet.
# Clean the output base manually to ensure build correctness
bazel clean
output_base=$(bazel info output_base)
bazel shutdown
# Sleep 5s to wait for jvm shutdown completely
# otherwise rm will fail with device or resource busy error
sleep 5
rm -rf ${output_base}

export TF_NEED_CUDA=0
echo "" | ./configure

BUILD_OPTS='-c opt --cpu=x64_windows_msvc --host_cpu=x64_windows_msvc --copt=/w --verbose_failures --experimental_ui'

bazel build $BUILD_OPTS tensorflow/tools/pip_package:build_pip_package || exit $?

./bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD

