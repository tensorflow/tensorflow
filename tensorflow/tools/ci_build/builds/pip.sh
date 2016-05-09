#!/usr/bin/env bash
# Copyright 2016 Google Inc. All Rights Reserved.
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
# Build the Python PIP installation package for TensorFlow and install
# the package.
# The PIP installation is done using the --user flag.
#
# Usage:
#   pip.sh CONTAINER_TYPE [--mavx] [--mavx2]
#                         [--test_tutorials] [--integration_tests]
#
# When executing the Python unit tests, the script obeys the shell
# variables: TF_BUILD_BAZEL_CLEAN, TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES,
# TF_BUILD_NO_CACHING_VIRTUALENV, NO_TEST_ON_INSTALL
#
# TF_BUILD_BAZEL_CLEAN, if set to any non-empty and non-0 value, directs the
# script to perform bazel clean prior to main build and test steps.
#
# TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES overrides the default extra pip packages
# to be installed in virtualenv before test_installation.sh is called. Multiple
# pakcage names are separated with spaces.
#
# TF_BUILD_NO_CACHING_VIRTUALENV: If set to any non-empty and non-0 value,
# will cause the script to force remove any existing (cached) virtualenv
# directory.
#
# If NO_TEST_ON_INSTALL has any non-empty and non-0 value, the test-on-install
# part will be skipped.
#
# If NO_TEST_USER_OPS has any non-empty and non-0 value, the testing of user-
# defined ops against the installation will be skipped.
#
# Use --mavx or --mavx2 to let bazel use --copt=-mavx or --copt=-mavx2 options
# while building the pip package, respectively.
#
# If the --test_tutorials flag is set, it will cause the script to run the
# tutorial tests (see test_tutorials.sh) after the PIP
# installation and the Python unit tests-on-install step. Likewise,
# --integration_tests will cause the integration tests (integration_tests.sh)
# to run.
#

INSTALL_EXTRA_PIP_PACKAGES=${TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES}


# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"


# Get the command line arguments
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )

if [[ ! -z "${TF_BUILD_BAZEL_CLEAN}" ]] && \
   [[ "${TF_BUILD_BAZEL_CLEAN}" != "0" ]]; then
  echo "TF_BUILD_BAZEL_CLEAN=${TF_BUILD_BAZEL_CLEAN}: Performing 'bazel clean'"
  bazel clean
fi

DO_TEST_USER_OPS=1
if [[ ! -z "${NO_TEST_USER_OPS}" ]] && \
   [[ "${NO_TEST_USER_OPS}" != "0" ]]; then
  echo "NO_TEST_USER_OPS=${NO_TEST_USER_OPS}: Will skip testing of user ops"
  DO_TEST_USER_OPS=0
fi

DO_TEST_TUTORIALS=0
DO_INTEGRATION_TESTS=0
MAVX_FLAG=""
while true; do
  if [[ "${1}" == "--test_tutorials" ]]; then
    DO_TEST_TUTORIALS=1
  elif [[ "${1}" == "--integration_tests" ]]; then
    DO_INTEGRATION_TESTS=1
  elif [[ "${1}" == "--mavx" ]]; then
    MAVX_FLAG="--copt=-mavx"
  elif [[ "${1}" == "--mavx2" ]]; then
    MAVX_FLAG="--copt=-mavx2"
  fi

  shift
  if [[ -z "${1}" ]]; then
    break
  fi
done

if [[ ! -z "${MAVX_FLAG}" ]]; then
  echo "Using MAVX flag: ${MAVX_FLAG}"
fi

PIP_BUILD_TARGET="//tensorflow/tools/pip_package:build_pip_package"
GPU_FLAG=""
if [[ ${CONTAINER_TYPE} == "cpu" ]]; then
  bazel build -c opt ${MAVX_FLAG} ${PIP_BUILD_TARGET} || \
      die "Build failed."
elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  bazel build -c opt --config=cuda ${MAVX_FLAG} ${PIP_BUILD_TARGET} || \
      die "Build failed."
  GPU_FLAG="--gpu"
else
  die "Unrecognized container type: \"${CONTAINER_TYPE}\""
fi

# If still in a virtualenv, deactivate it first
if [[ ! -z "$(which deactivate)" ]]; then
  echo "It appears that we are already in a virtualenv. Deactivating..."
  deactivate || die "FAILED: Unable to deactivate from existing virtualenv"
fi

# Obtain the path to Python binary
source tools/python_bin_path.sh

# Assume: PYTHON_BIN_PATH is exported by the script above
if [[ -z "$PYTHON_BIN_PATH" ]]; then
  die "PYTHON_BIN_PATH was not provided. Did you run configure?"
fi

# Determine the major and minor versions of Python being used (e.g., 2.7)
# This info will be useful for determining the directory of the local pip
# installation of Python
PY_MAJOR_MINOR_VER=$(${PYTHON_BIN_PATH} -V 2>&1 | awk '{print $NF}' | cut -d. -f-2)

echo "Python binary path to be used in PIP install: ${PYTHON_BIN_PATH} "\
"(Major.Minor version: ${PY_MAJOR_MINOR_VER})"

# Build PIP Wheel file
PIP_TEST_ROOT="pip_test"
PIP_WHL_DIR="${PIP_TEST_ROOT}/whl"
PIP_WHL_DIR=$(realpath ${PIP_WHL_DIR})  # Get absolute path
rm -rf ${PIP_WHL_DIR} && mkdir -p ${PIP_WHL_DIR}
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} || \
    die "build_pip_package FAILED"

# Perform installation
WHL_PATH=$(ls ${PIP_WHL_DIR}/tensorflow*.whl)
if [[ $(echo ${WHL_PATH} | wc -w) -ne 1 ]]; then
  die "ERROR: Failed to find exactly one built TensorFlow .whl file in "\
"directory: ${PIP_WHL_DIR}"
fi

echo "whl file path = ${WHL_PATH}"

# Install, in user's local home folder
echo "Installing pip whl file: ${WHL_PATH}"

# Create virtualenv directory for install test
VENV_DIR="${PIP_TEST_ROOT}/venv"
if [[ -d "${VENV_DIR}" ]] &&
   [[ ! -z "${TF_BUILD_NO_CACHING_VIRTUALENV}" ]] &&
   [[ "${TF_BUILD_NO_CACHING_VIRTUALENV}" != "0" ]]; then
  echo "TF_BUILD_NO_CACHING_VIRTUALENV=${TF_BUILD_NO_CACHING_VIRTUALENV}:"
  echo "Removing existing virtualenv directory: ${VENV_DIR}"

  rm -rf "${VENV_DIR}" || \
      die "Failed to remove existing virtualenv directory: ${VENV_DIR}"
fi

mkdir -p ${VENV_DIR} || \
    die "FAILED to create virtualenv directory: ${VENV_DIR}"

# Verify that virtualenv exists
if [[ -z $(which virtualenv) ]]; then
  die "FAILED: virtualenv not available on path"
fi

virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" "${VENV_DIR}" || \
    die "FAILED: Unable to create virtualenv"

source "${VENV_DIR}/bin/activate" || \
    die "FAILED: Unable to activate virtualenv"


# Install the pip file in virtual env (plus missing dependencies)
pip install -v ${WHL_PATH} || die "pip install (without --upgrade) FAILED"
# Force tensorflow reinstallation. Otherwise it may not get installed from
# last build if it had the same version number as previous build.
pip install -v --upgrade --no-deps --force-reinstall ${WHL_PATH} || \
    die "pip install (forcing to reinstall tensorflow) FAILED"
echo "Successfully installed pip package ${WHL_PATH}"

# Install extra pip packages required by the test-on-install
for PACKAGE in ${INSTALL_EXTRA_PIP_PACKAGES}; do
  echo "Installing extra pip package required by test-on-install: ${PACKAGE}"

  pip install ${PACKAGE} || \
      die "pip install ${PACKAGE} FAILED"
done

# If NO_TEST_ON_INSTALL is set to any non-empty value, skip all Python
# tests-on-install and exit right away
if [[ ! -z "${NO_TEST_ON_INSTALL}" ]] &&
   [[ "${NO_TEST_ON_INSTALL}" != "0" ]]; then
  echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
  echo "  Skipping ALL Python unit tests on install"
  exit 0
fi

# Call test_installation.sh to perform test-on-install

"${SCRIPT_DIR}/test_installation.sh" --virtualenv ${GPU_FLAG} ||
    die "PIP tests-on-install FAILED"

# Test user ops
if [[ "${DO_TEST_USER_OPS}" == "1" ]]; then
  "${SCRIPT_DIR}/test_user_ops.sh" --virtualenv ${GPU_FLAG} || \
      die "PIP user-op tests-on-install FAILED"
fi

# Optional: Run the tutorial tests
if [[ "${DO_TEST_TUTORIALS}" == "1" ]]; then
  "${SCRIPT_DIR}/test_tutorials.sh" --virtualenv || \
      die "PIP tutorial tests-on-install FAILED"
fi

# Optional: Run integration tests
if [[ "${DO_INTEGRATION_TESTS}" == "1" ]]; then
  "${SCRIPT_DIR}/integration_tests.sh" --virtualenv || \
      die "Integration tests on install FAILED"
fi

deactivate || \
    die "FAILED: Unable to deactivate virtualenv"
