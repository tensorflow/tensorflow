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
source "tensorflow/tools/ci_build/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# load bazel_test_lib.sh
source "tensorflow/tools/ci_build/windows/bazel/bazel_test_lib.sh" \
  || { echo "Failed to source bazel_test_lib.sh" >&2; exit 1; }

clean_output_base

run_configure_for_cpu_build

bazel build $BUILD_OPTS tensorflow/tools/pip_package:build_pip_package || exit $?

rm -f ./tensorflow-*.whl
./bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD

# Running python tests on Windows needs pip package installed
echo "y" | pip uninstall tensorflow -q || true
PIP_NAME=$(ls tensorflow-*.whl)
pip install ${PIP_NAME}

# Create a python test directory to avoid package name conflict
PY_TEST_DIR="py_test_dir"
rm -rf "${PY_TEST_DIR}"
mkdir -p "${PY_TEST_DIR}"
cmd /c "mklink /J ${PY_TEST_DIR}\\tensorflow .\\tensorflow"

failing_cpu_py_tests=$(get_failing_cpu_py_tests ${PY_TEST_DIR})

passing_tests=$(bazel query "kind(py_test,  //${PY_TEST_DIR}/tensorflow/python/...) - (${failing_cpu_py_tests})" |
  # We need to strip \r so that the result could be store into a variable under MSYS
  tr '\r' ' ')

BUILD_OPTS='-c opt --cpu=x64_windows_msvc --host_cpu=x64_windows_msvc --copt=/w --verbose_failures --experimental_ui'
# Define no_tensorflow_py_deps=true so that every py_test has no deps anymore,
# which will result testing system installed tensorflow
bazel test $BUILD_OPTS -k $passing_tests --define=no_tensorflow_py_deps=true --test_output=errors


