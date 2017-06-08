#!/usr/bin/env bash
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
#
# ==============================================================================
#
# Usage:
#   This script assumes it is invoked while `pwd` points to the repository root.
#   ./tensorflow/tools/ci_build/mac/gpu/pip/run.sh
#
#   By default, script will use python2.
#   You can elect to use python 3 by setting TF_PYTHON_VERSION environment
#   variable to "3".
#

# Mac seems to be losing the environment variables, remind the important ones:
export PATH="/usr/local/cuda/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export LB_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:"

# Enable stop on errors, and debugging.
set -e
set -x

# This script is under <repo_root>/tensorflow/tools/ci_build/mac/gpu/pip/
# Save repository root directory for future use.
REPO_ROOT=`pwd`
# Get common small bash utilities.
source "${REPO_ROOT}/tensorflow/tools/ci_build/builds/builds_common.sh"

# Pick which python version to use. Will be fed into configure script.
TF_PYTHON_VERSION=${TF_PYTHON_VERSION:-2}
if [[ ${TF_PYTHON_VERSION} == "2" ]]; then
  export PYTHON_BIN_PATH="/usr/local/bin/python"
elif [[ ${TF_PYTHON_VERSION} == "3" ]]; then
  export PYTHON_BIN_PATH="/usr/local/bin/python3"
else
  die "Invalid value set for TF_PYTHON_VERSION, 2 or 3 expected, value: ${TF_PYTHON_VERSION}"
fi

# Run configure script.
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_CUDA=1
export TF_NEED_OPENCL=0
yes "" | ./configure

# Build the pip package builder script.
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

# Common directory to put all pip test stuff under.
PIP_TEST_ROOT="pip_test"

# A directory for pip package.
PIP_WHL_DIR="${PIP_TEST_ROOT}/whl"
PIP_WHL_DIR=$(realpath ${PIP_WHL_DIR})  # Get absolute path
rm -rf ${PIP_WHL_DIR} && mkdir -p ${PIP_WHL_DIR}

# Create the pip package by running the above built script.
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} || \
  die "build_pip_package FAILED"

# Create a virtualenv
VENV_DIR="${PIP_TEST_ROOT}/venv"
rm -rf "${VENV_DIR}" && mkdir -p ${VENV_DIR}
virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" "${VENV_DIR}" || \
  die "FAILED: Unable to create virtualenv"
source "${VENV_DIR}/bin/activate" || \
  die "FAILED: Unable to activate virtualenv"

# TODO(gunan): Upgrade all pip versions to over 8.1.2 on all Jenkins machines.
# Upgrade pip so it supports tags such as cp27mu, manylinux1 etc.
echo "Upgrade pip in virtualenv"
pip install --upgrade pip==8.1.2

pip install -v --upgrade --force-reinstall --no-deps ${PIP_WHL_DIR}/*.whl || \
  die "pip install FAILED."

# Run unit tests.
"${REPO_ROOT}/tensorflow/tools/ci_build/builds/test_installation.sh" --virtualenv --gpu --mac || \
  die "PIP tests-on-install FAILED"

# Run tutorial tests.
"${REPO_ROOT}/tensorflow/tools/ci_build/builds/test_user_ops.sh" --virtualenv --gpu || \
  die "PIP tutorial tests-on-install FAILED"

# Run integration tests.
"${REPO_ROOT}/tensorflow/tools/ci_build/builds/integration_tests.sh" --virtualenv || \
  die "Integration tests on install FAILED"

# Deactivate the virtualenv and exit.
deactivate
