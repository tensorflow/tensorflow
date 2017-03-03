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
# ==============================================================================
#
# Build the Python PIP installation package for TensorFlow and install
# the package.
# The PIP installation is done using the --user flag.
#
# Usage:
#   pip.sh CONTAINER_TYPE [--test_tutorials] [--integration_tests] [bazel flags]
#
# When executing the Python unit tests, the script obeys the shell
# variables: TF_BUILD_BAZEL_CLEAN, TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES,
# NO_TEST_ON_INSTALL
#
# TF_BUILD_BAZEL_CLEAN, if set to any non-empty and non-0 value, directs the
# script to perform bazel clean prior to main build and test steps.
#
# TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES overrides the default extra pip packages
# to be installed in virtualenv before run_pip_tests.sh is called. Multiple
# pakcage names are separated with spaces.
#
# If NO_TEST_ON_INSTALL has any non-empty and non-0 value, the test-on-install
# part will be skipped.
#
# If NO_TEST_USER_OPS has any non-empty and non-0 value, the testing of user-
# defined ops against the installation will be skipped.
#
# Any flags not listed in the usage above will be passed directly to Bazel.
#
# If the --test_tutorials flag is set, it will cause the script to run the
# tutorial tests (see test_tutorials.sh) after the PIP
# installation and the Python unit tests-on-install step. Likewise,
# --integration_tests will cause the integration tests (integration_tests.sh)
# to run.
#

# Helper function: Strip leading and trailing whitespaces
str_strip () {
  echo -e "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Fixed naming patterns for wheel (.whl) files given different python versions
if [[ $(uname) == "Linux" ]]; then
  declare -A WHL_TAGS
  WHL_TAGS=(["2.7"]="cp27-none" ["3.4"]="cp34-cp34m" ["3.5"]="cp35-cp35m")
fi


INSTALL_EXTRA_PIP_PACKAGES=${TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES}


# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"


# Get the command line arguments
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift

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
BAZEL_FLAGS=""
while true; do
  if [[ "${1}" == "--test_tutorials" ]]; then
    DO_TEST_TUTORIALS=1
  elif [[ "${1}" == "--integration_tests" ]]; then
    DO_INTEGRATION_TESTS=1
  else
    BAZEL_FLAGS="${BAZEL_FLAGS} ${1}"
  fi

  shift
  if [[ -z "${1}" ]]; then
    break
  fi
done

BAZEL_FLAGS=$(str_strip "${BAZEL_FLAGS}")

echo "Using Bazel flags: ${BAZEL_FLAGS}"

PIP_BUILD_TARGET="//tensorflow/tools/pip_package:build_pip_package"
GPU_FLAG=""
if [[ ${CONTAINER_TYPE} == "cpu" ]] || \
   [[ ${CONTAINER_TYPE} == "debian.jessie.cpu" ]]; then
  bazel build ${BAZEL_FLAGS} ${PIP_BUILD_TARGET} || \
      die "Build failed."
elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  bazel build ${BAZEL_FLAGS} ${PIP_BUILD_TARGET} || \
      die "Build failed."
  GPU_FLAG="--gpu"
else
  die "Unrecognized container type: \"${CONTAINER_TYPE}\""
fi

MAC_FLAG=""
if [[ $(uname) == "Darwin" ]]; then
  MAC_FLAG="--mac"
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
if [[ -z "${PY_MAJOR_MINOR_VER}" ]]; then
  die "ERROR: Unable to determine the major.minor version of Python"
fi

echo "Python binary path to be used in PIP install: ${PYTHON_BIN_PATH} "\
"(Major.Minor version: ${PY_MAJOR_MINOR_VER})"

# Build PIP Wheel file
PIP_TEST_ROOT="pip_test"
PIP_WHL_DIR="${PIP_TEST_ROOT}/whl"
PIP_WHL_DIR=$(realpath ${PIP_WHL_DIR})  # Get absolute path
rm -rf ${PIP_WHL_DIR} && mkdir -p ${PIP_WHL_DIR}
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} ${GPU_FLAG} || \
    die "build_pip_package FAILED"

WHL_PATH=$(ls ${PIP_WHL_DIR}/tensorflow*.whl)
if [[ $(echo ${WHL_PATH} | wc -w) -ne 1 ]]; then
  die "ERROR: Failed to find exactly one built TensorFlow .whl file in "\
"directory: ${PIP_WHL_DIR}"
fi

# Rename the whl file properly so it will have the python
# version tags and platform tags that won't cause pip install issues.
if [[ $(uname) == "Linux" ]]; then
  PY_TAGS=${WHL_TAGS[${PY_MAJOR_MINOR_VER}]}
  PLATFORM_TAG=$(to_lower "$(uname)_$(uname -m)")
# MAC has bash v3, which does not have associative array
elif [[ $(uname) == "Darwin" ]]; then
  if [[ ${PY_MAJOR_MINOR_VER} == "2.7" ]]; then
    PY_TAGS="py2-none"
  elif [[ ${PY_MAJOR_MINOR_VER} == "3.5" ]]; then
    PY_TAGS="py3-none"
  fi
  PLATFORM_TAG="any"
fi

WHL_DIR=$(dirname "${WHL_PATH}")
WHL_BASE_NAME=$(basename "${WHL_PATH}")

if [[ ! -z "${PY_TAGS}" ]]; then
  NEW_WHL_BASE_NAME=$(echo ${WHL_BASE_NAME} | cut -d \- -f 1)-\
$(echo ${WHL_BASE_NAME} | cut -d \- -f 2)-${PY_TAGS}-${PLATFORM_TAG}.whl

  if [[ ! -f "${WHL_DIR}/${NEW_WHL_BASE_NAME}" ]]; then
    cp "${WHL_DIR}/${WHL_BASE_NAME}" "${WHL_DIR}/${NEW_WHL_BASE_NAME}" && \
      echo "Copied wheel file: ${WHL_BASE_NAME} --> ${NEW_WHL_BASE_NAME}" || \
      die "ERROR: Failed to copy wheel file to ${NEW_WHL_BASE_NAME}"
  fi
fi

if [[ $(uname) == "Linux" ]]; then
  AUDITED_WHL_NAME="${WHL_DIR}/$(echo ${WHL_BASE_NAME} | sed "s/linux/manylinux1/")"

  # Repair the wheels for cpu manylinux1
  if [[ ${CONTAINER_TYPE} == "cpu" ]]; then
    echo "auditwheel repairing ${WHL_PATH}"
    auditwheel repair -w ${WHL_DIR} ${WHL_PATH}

    if [[ -f ${AUDITED_WHL_NAME} ]]; then
      WHL_PATH=${AUDITED_WHL_NAME}
      echo "Repaired manylinx1 wheel file at: ${WHL_PATH}"
    else
      die "ERROR: Cannot find repaired wheel."
    fi
  # Copy and rename for gpu manylinux as we do not want auditwheel to package in libcudart.so
  elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
    WHL_PATH=${AUDITED_WHL_NAME}
    cp ${WHL_DIR}/${WHL_BASE_NAME} ${WHL_PATH}
    echo "Copied manylinx1 wheel file at ${WHL_PATH}"
  fi
fi

# Perform installation
echo "Installing pip whl file: ${WHL_PATH}"

# Create virtualenv directory for install test
VENV_DIR="${PIP_TEST_ROOT}/venv"

if [[ -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}" && \
      echo "Removed existing virtualenv directory: ${VENV_DIR}" || \
      die "Failed to remove existing virtualenv directory: ${VENV_DIR}"
fi

mkdir -p ${VENV_DIR} && \
    echo "Created virtualenv directory: ${VENV_DIR}" || \
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

# Upgrade pip so it supports tags such as cp27mu, manylinux1 etc.
echo "Upgrade pip in virtualenv"
pip install --upgrade pip==8.1.2

# Force tensorflow reinstallation. Otherwise it may not get installed from
# last build if it had the same version number as previous build.
PIP_FLAGS="--upgrade --force-reinstall --no-deps"
pip install -v ${PIP_FLAGS} ${WHL_PATH} || \
    die "pip install (forcing to reinstall tensorflow) FAILED"
echo "Successfully installed pip package ${WHL_PATH}"

# Install extra pip packages required by the test-on-install
for PACKAGE in ${INSTALL_EXTRA_PIP_PACKAGES}; do
  echo "Installing extra pip package required by test-on-install: ${PACKAGE}"

  pip install ${PACKAGE} || \
      die "pip install ${PACKAGE} FAILED"
done

if [[ ! -z "${NO_TEST_ON_INSTALL}" ]] &&
   [[ "${NO_TEST_ON_INSTALL}" != "0" ]]; then
  echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
  echo "  Skipping ALL Python unit tests on install"
else
  # Call run_pip_tests.sh to perform test-on-install
  "${SCRIPT_DIR}/run_pip_tests.sh" --virtualenv ${GPU_FLAG} ${MAC_FLAG} ||
      die "PIP tests-on-install FAILED"
fi

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
