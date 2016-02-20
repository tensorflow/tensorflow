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

# Build the Python PIP installation package for TensorFlow
# and run the Python unit tests from the source code on the installation
#
# Usage:
#   pip.sh CONTAINER_TYPE
#
# When executing the Python unit tests, the script obeys the shell
# variables: PY_TEST_WHITELIST, PY_TEST_BLACKLIST, PY_TEST_GPU_BLACKLIST,
# TF_BUILD_BAZEL_CLEAN, NO_TEST_ON_INSTALL
#
# To select only a subset of the Python tests to run, set the environment
# variable PY_TEST_WHITELIST, e.g.,
#   PY_TEST_WHITELIST="tensorflow/python/kernel_tests/shape_ops_test.py"
# Separate the tests with a colon (:). Leave this environment variable empty
# to disable the whitelist.
#
# You can also ignore a set of the tests by using the environment variable
# PY_TEST_BLACKLIST. For example, you can include in PY_TEST_BLACKLIST the
# tests that depend on Python modules in TensorFlow source that are not
# exported publicly.
#
# In addition, you can put blacklist for only GPU build inthe environment
# variable PY_TEST_GPU_BLACKLIST.
#
# TF_BUILD_BAZEL_CLEAN, if set to any non-empty and non-0 value, directs the
# script to perform bazel clean prior to main build and test steps.
#
# If the environmental variable NO_TEST_ON_INSTALL is set to any non-empty
# value, the script will exit after the pip install step.

# =============================================================================
# Test blacklist: General
#
# tensorflow/python/framework/ops_test.py
#   depends on depends on "test_ops", which is defined in a C++ file wrapped as
#   a .py file through the Bazel rule “tf_gen_ops_wrapper_py”.
# tensorflow/util/protobuf/compare_test.py:
#   depends on compare_test_pb2 defined outside Python
# tensorflow/python/framework/device_test.py:
#   depends on CheckValid() and ToString(), both defined externally
#
PY_TEST_BLACKLIST="${PY_TEST_BLACKLIST}:"\
"tensorflow/python/framework/ops_test.py:"\
"tensorflow/python/util/protobuf/compare_test.py:"\
"tensorflow/python/framework/device_test.py"

# Test blacklist: GPU-only
PY_TEST_GPU_BLACKLIST="${PY_TEST_GPU_BLACKLIST}:"\
"tensorflow/python/framework/function_test.py"

# =============================================================================

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

echo "Python binary path to be used in PIP install-test: ${PYTHON_BIN_PATH} "\
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
${PYTHON_BIN_PATH} -m pip install -v --user ${WHL_PATH} \
|| die "pip install (without --upgrade) FAILED"

# If NO_TEST_ON_INSTALL is set to any non-empty value, skip all Python
# tests-on-install and exit right away
if [[ ! -z ${NO_TEST_ON_INSTALL} ]]; then
  echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
  echo "  Skipping ALL Python unit tests on install"
  exit 0
fi

# Directory from which the unit-test files will be run
PY_TEST_DIR_REL="pip_test/tests"
PY_TEST_DIR=$(abs_path ${PY_TEST_DIR_REL})  # Get absolute path
rm -rf ${PY_TEST_DIR} && mkdir -p ${PY_TEST_DIR}

# Create test log directory
PY_TEST_LOG_DIR_REL=${PY_TEST_DIR_REL}/logs
PY_TEST_LOG_DIR=$(abs_path ${PY_TEST_LOG_DIR_REL})  # Absolute path

mkdir ${PY_TEST_LOG_DIR}

# Copy source files that are required by the tests but are not included in the
# PIP package

# Look for local Python library directory
LIB_PYTHON_DIR=""

# Candidate locations of the local Python library directory
LIB_PYTHON_DIR_CANDS="${HOME}/.local/lib/python${PY_MAJOR_MINOR_VER}* "\
"${HOME}/Library/Python/${PY_MAJOR_MINOR_VER}*/lib/python"

for CAND in ${LIB_PYTHON_DIR_CANDS}; do
  if [[ -d "${CAND}" ]]; then
    LIB_PYTHON_DIR="${CAND}"
    break
  fi
done

if [[ -z ${LIB_PYTHON_DIR} ]]; then
  die "Failed to find local Python library directory"
else
  echo "Found local Python library directory at: ${LIB_PYTHON_DIR}"
fi

PACKAGES_DIR=$(ls -d ${LIB_PYTHON_DIR}/*-packages | head -1)

echo "Copying some source directories that are required by tests but are "\
"not included in install to Python packages directory: ${PACKAGES_DIR}"

# Files for tensorflow.python.tools
rm -rf ${PACKAGES_DIR}/tensorflow/python/tools
cp -r tensorflow/python/tools \
      ${PACKAGES_DIR}/tensorflow/python/tools
touch ${PACKAGES_DIR}/tensorflow/python/tools/__init__.py  # Make module visible

# Files for tensorflow.examples
rm -rf ${PACKAGES_DIR}/tensorflow/examples
mkdir -p ${PACKAGES_DIR}/tensorflow/examples/image_retraining
cp -r tensorflow/examples/image_retraining/retrain.py \
      ${PACKAGES_DIR}/tensorflow/examples/image_retraining/retrain.py
touch ${PACKAGES_DIR}/tensorflow/examples/__init__.py
touch ${PACKAGES_DIR}/tensorflow/examples/image_retraining/__init__.py

echo "Copying additional files required by tests to working directory "\
"for test: ${PY_TEST_DIR}"

# Image files required by some tests, e.g., images_ops_test.py
mkdir -p ${PY_TEST_DIR}/tensorflow/core/lib
rm -rf ${PY_TEST_DIR}/tensorflow/core/lib/jpeg
cp -r tensorflow/core/lib/jpeg ${PY_TEST_DIR}/tensorflow/core/lib
rm -rf ${PY_TEST_DIR}/tensorflow/core/lib/png
cp -r tensorflow/core/lib/png ${PY_TEST_DIR}/tensorflow/core/lib

# Run tests
DIR0=$(pwd)
ALL_PY_TESTS=$(find tensorflow/{contrib,examples,models,python,tensorboard} -name "*_test.py" | sort)
# TODO(cais): Add tests in tensorflow/contrib

PY_TEST_COUNT=$(echo ${ALL_PY_TESTS} | wc -w)

if [[ ${PY_TEST_COUNT} -eq 0 ]]; then
  die "ERROR: Cannot find any tensorflow Python unit tests to run on install"
fi

# Iterate through all the Python unit test files using the installation
COUNTER=0
PASS_COUNTER=0
FAIL_COUNTER=0
SKIP_COUNTER=0
FAILED_TESTS=""
FAILED_TEST_LOGS=""

for TEST_FILE_PATH in ${ALL_PY_TESTS}; do
  ((COUNTER++))

  PROG_STR="(${COUNTER} / ${PY_TEST_COUNT})"

  # If PY_TEST_WHITELIST is not empty, only the white-listed tests will be run
  if [[ ! -z ${PY_TEST_WHITELIST} ]] && \
     [[ ! ${PY_TEST_WHITELIST} == *"${TEST_FILE_PATH}"* ]]; then
    ((SKIP_COUNTER++))
    echo "${PROG_STR} Non-whitelisted test SKIPPED: ${TEST_FILE_PATH}"
    continue
  fi

  # If the test is in the black list, skip it
  if [[ ${PY_TEST_BLACKLIST} == *"${TEST_FILE_PATH}"* ]]; then
    ((SKIP_COUNTER++))
    echo "${PROG_STR} Blacklisted test SKIPPED: ${TEST_FILE_PATH}"
    continue
  fi

  # Copy to a separate directory to guard against the possibility of picking up
  # modules in the source directory
  cp ${TEST_FILE_PATH} ${PY_TEST_DIR}/

  TEST_BASENAME=$(basename "${TEST_FILE_PATH}")

  # Relative path of the test log. Use long path in case there are duplicate
  # file names in the Python tests
  TEST_LOG_REL="${PY_TEST_LOG_DIR_REL}/${TEST_FILE_PATH}.log"
  mkdir -p $(dirname ${TEST_LOG_REL})  # Create directory for log

  TEST_LOG=$(abs_path ${TEST_LOG_REL})  # Absolute path

  # Before running the test, cd away from the Tensorflow source to
  # avoid the possibility of picking up dependencies from the
  # source directory
  cd ${PY_TEST_DIR}
  ${PYTHON_BIN_PATH} ${PY_TEST_DIR}/${TEST_BASENAME} >${TEST_LOG} 2>&1

  # Check for pass or failure status of the test outtput and exit
  if [[ $? -eq 0 ]]; then
    ((PASS_COUNTER++))

    echo "${PROG_STR} Python test-on-install PASSED: ${TEST_FILE_PATH}"
  else
    ((FAIL_COUNTER++))

    FAILED_TESTS="${FAILED_TESTS} ${TEST_FILE_PATH}"

    FAILED_TEST_LOGS="${FAILED_TEST_LOGS} ${TEST_LOG_REL}"

    echo "${PROG_STR} Python test-on-install FAILED: ${TEST_FILE_PATH}"
    echo "  Log @: ${TEST_LOG_REL}"
    echo "============== BEGINS failure log content =============="
    cat ${TEST_LOG}
    echo "============== ENDS failure log content =============="
    echo ""
  fi
  cd ${DIR0}

  # Clean up files for this test
  rm -f ${PY_TEST_DIR}/${TEST_BASENAME}

done

echo ""
echo "${PY_TEST_COUNT} Python test(s):" \
     "${PASS_COUNTER} passed;" \
     "${FAIL_COUNTER} failed; " \
     "${SKIP_COUNTER} skipped"
echo "Test logs directory: ${PY_TEST_LOG_DIR_REL}"

if [[ ${FAIL_COUNTER} -eq 0  ]]; then
  echo ""
  echo "Python test-on-install SUCCEEDED"

  exit 0
else
  echo "FAILED test(s):"
  FAILED_TEST_LOGS=($FAILED_TEST_LOGS)
  FAIL_COUNTER=0
  for TEST_NAME in ${FAILED_TESTS}; do
    echo "  ${TEST_NAME} (Log @: ${FAILED_TEST_LOGS[${FAIL_COUNTER}]})"
    ((FAIL_COUNTER++))
  done

  echo ""
  echo "Python test-on-install FAILED"
  exit 1
fi
