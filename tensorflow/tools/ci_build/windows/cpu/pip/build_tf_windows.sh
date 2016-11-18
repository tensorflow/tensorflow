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
cd ${script_dir%%tensorflow/tools/ci_build/windows/cpu/pip}

# Use a temporary directory with a short name.
export TMPDIR="C:/tmp"

# Set bash path
export BAZEL_SH="C:/tools/msys64/usr/bin/bash"

# Set Python path for ./configure
export PYTHON_BIN_PATH="C:/Program Files/Anaconda3/python"

# Set Python path for cc_configure.bzl
export BAZEL_PYTHON="C:/Program Files/Anaconda3/python"

# Set Visual Studio path
export BAZEL_VS="C:/Program Files (x86)/Microsoft Visual Studio 14.0"

# Add python into PATH, it's needed because gen_git_source.py uses
# '/usr/bin/env python' as a shebang
export PATH="/c/Program Files/Anaconda3:$PATH"

export TF_NEED_CUDA=0

# bazel clean --expunge doesn't work on Windows yet.
# Clean the output base manually to ensure build correctness
bazel clean
output_base=$(bazel info output_base)
bazel shutdown
# Sleep 5s to wait for jvm shutdown completely
# otherwise rm will fail with device or resource busy error
sleep 5
rm -rf ${output_base}

echo "" | ./configure

bazel build -c opt --cpu=x64_windows_msvc --host_cpu=x64_windows_msvc\
    --copt="/w" --verbose_failures --experimental_ui\
      tensorflow/tools/pip_package:build_pip_package || exit $?


./bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD
