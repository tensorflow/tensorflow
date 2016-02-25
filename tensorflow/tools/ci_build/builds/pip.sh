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
#   pip.sh CONTAINER_TYPE
#
# When executing the Python unit tests, the script obeys the shell
# variables: TF_BUILD_BAZEL_CLEAN, NO_TEST_ON_INSTALL.
#
# TF_BUILD_BAZEL_CLEAN, if set to any non-empty and non-0 value, directs the
# script to perform bazel clean prior to main build and test steps.
#
# If NO_TEST_ON_INSTALL has any non-empty and non-0 value, the test-on-install
# part will be skipped.


# Helper functions
# Get the absolute path from a path
abs_path() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}


# Exit after a failure
die() {
    echo $@
    exit 1
}

# Uninstall existing pip packages
uninstall_existing_pip_packages() {
  # Usage: uninstall_existing <PYTHON_BIN_PATH> <PY_MAJOR_MINOR_VER>
  PACKAGES_TO_UNINSTALL="protobuf tensorflow"

  # Candidate locations of the local Python library directory
  # Assumes that the TensorFlow pip installation is done using the "--user" flag
  LIB_PYTHON_DIR_CANDS="${HOME}/.local/lib/python${2}* "\
"${HOME}/Library/Python/${2}*/lib/python"

  for CAND in ${LIB_PYTHON_DIR_CANDS}; do
    if [[ -d "${CAND}" ]]; then
      LIB_PYTHON_DIR="${CAND}"
      break
    fi
  done

  if [[ -z ${LIB_PYTHON_DIR} ]]; then
    echo "uninstall_existing_pip_packages: "\
"No local Python library directory is found. Moving on."
    return 0
  else
    echo "Found local Python library directory at: ${LIB_PYTHON_DIR}"
  fi

  PACKAGES_DIR=$(ls -d ${LIB_PYTHON_DIR}/*-packages | head -1)

  for PACKAGE in ${PACKAGES_TO_UNINSTALL}; do
    if [[ -d "${PACKAGES_DIR}/${PACKAGE}" ]]; then
      echo "Uninstalling existing local installation of ${PACKAGE}"

      "$1" -m pip uninstall -y "${PACKAGE}"
      if [[ $? != 0 ]]; then
        echo "FAILED to uninstall existing local installation of ${PACKAGE}"
        return 1
      fi
    fi
  done

  return 0
}


# Get the command line arguments
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )

if [[ ! -z "${TF_BUILD_BAZEL_CLEAN}" ]] && \
   [[ "${TF_BUILD_BAZEL_CLEAN}" != "0" ]]; then
  echo "TF_BUILD_BAZEL_CLEAN=${TF_BUILD_BAZEL_CLEAN}: Performing 'bazel clean'"
  bazel clean
fi

PIP_BUILD_TARGET="//tensorflow/tools/pip_package:build_pip_package"
if [[ ${CONTAINER_TYPE} == "cpu" ]]; then
  bazel build -c opt ${PIP_BUILD_TARGET} || die "Build failed."
elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  bazel build -c opt --config=cuda ${PIP_BUILD_TARGET} || die "Build failed."
else
  die "Unrecognized container type: \"${CONTAINER_TYPE}\""
fi

echo "PY_TEST_WHITELIST: ${PY_TEST_WHITELIST}"
echo "PY_TEST_BLACKLIST: ${PY_TEST_BLACKLIST}"
echo "PY_TEST_GPU_BLACKLIST: ${PY_TEST_GPU_BLACKLIST}"

# Append GPU-only test blacklist
if [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  PY_TEST_BLACKLIST="${PY_TEST_BLACKLIST}:${PY_TEST_GPU_BLACKLIST}"
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
PIP_WHL_DIR="pip_test/whl"
PIP_WHL_DIR=$(abs_path ${PIP_WHL_DIR})  # Get absolute path
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

# Call pip install on the whl file. We are doing it without the --upgrade
# option. So dependency updates will need to be performed separately in
# the environment.
uninstall_existing_pip_packages "${PYTHON_BIN_PATH}" "${PY_MAJOR_MINOR_VER}"

"${PYTHON_BIN_PATH}" -m pip install -v --user ${WHL_PATH} \
&& echo "Successfully installed pip package ${WHL_PATH}" \
|| die "pip install (without --upgrade) FAILED"

# If NO_TEST_ON_INSTALL is set to any non-empty value, skip all Python
# tests-on-install and exit right away
if [[ ! -z "${NO_TEST_ON_INSTALL}" ]] &&
   [[ "${NO_TEST_ON_INSTALL}" != "0" ]]; then
  echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
  echo "  Skipping ALL Python unit tests on install"
  exit 0
fi

# Call pip_test.sh to perform test-on-install
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${DIR}/pip_test.sh"
