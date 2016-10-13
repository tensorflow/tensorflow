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
# Build the Python PIP installation package for TensorFlow
# and run the Python unit tests from the source code on the installation
#
# Usage:
#   test_installation.sh [--virtualenv] [--gpu]
#
# If the flag --virtualenv is set, the script will use "python" as the Python
# binary path. Otherwise, it will use tools/python_bin_path.sh to determine
# the Python binary path.
#
# The --gpu flag informs the script that this is a GPU build, so that the
# appropriate test blacklists can be applied accordingly.
#
# When executing the Python unit tests, the script obeys the shell
# variables: PY_TEST_WHITELIST, PY_TEST_BLACKLIST, PY_TEST_GPU_BLACKLIST,
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
# TF_BUILD_SERIAL_INSTALL_TESTS, if set to any non-empty and non-0 value,
# will force the Python install tests to run serially, overriding than the
# concurrent testing behavior.
#
# TF_BUILD_EXTRA_EXCLUSIVE_INSTALL_TESTS, add to the default list of
# Python unit tests to run in exclusive mode (i.e., not concurrently with
# other tests), separated with colons
#
# TF_BUILD_FILTER_INSTALL_TESTS_BY_TAG: If set to a non-empty string
# (e.g., "local"), will filter the Python install-tests by that string as
# bazel tags. Multiple filter tags can be used. Both the inclusive filtering
# mode and the exclusive filtering mode can be used. For example:
#
#   TF_BUILD_FILTER_INSTALL_TESTS_BY_TAG="local,-manual"
#
# will let the script run the Python unit tests that have the tag "local"
# and do not have the tag "manual". The "-" marks the exclusive filtering
# mode. The inclusive mode is the default. Use commas to separate the tags.
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
# tensorflow/python/framework/file_system_test.py:
#   depends on having the .so which is not shipped in the pip package.
# tensorflow/contrib/quantization/*:
#   These depend on an .so mechanism that's not shipped in the pip package.
# tensorflow/python/platform/default/*_test.py:
#   These are obsolete and replaced by corresponding files in python/platform.
#   They will be removed in the future.

PY_TEST_BLACKLIST="${PY_TEST_BLACKLIST}:"\
"tensorflow/python/framework/ops_test.py:"\
"tensorflow/python/util/protobuf/compare_test.py:"\
"tensorflow/python/framework/device_test.py:"\
"tensorflow/python/framework/file_system_test.py:"\
"tensorflow/contrib/quantization/python/dequantize_op_test.py:"\
"tensorflow/contrib/quantization/python/quantized_conv_ops_test.py:"\
"tensorflow/contrib/quantization/tools/quantize_graph_test.py:"\
"tensorflow/contrib/session_bundle/exporter_test.py:"\
"tensorflow/contrib/session_bundle/session_bundle_test.py:"\
"tensorflow/python/platform/default/_resource_loader_test.py:"\
"tensorflow/python/platform/default/flags_test.py:"\
"tensorflow/python/platform/default/logging_test.py:"\
"tensorflow/contrib/learn/nonlinear_test.py:"

# Test blacklist: GPU-only
PY_TEST_GPU_BLACKLIST="${PY_TEST_GPU_BLACKLIST}:"\
"tensorflow/python/client/session_test.py:"\
"tensorflow/python/framework/function_test.py:"\
"tensorflow/contrib/tensor_forest/python/kernel_tests/scatter_add_ndim_op_test.py"

# Tests that should be run in the exclusive mode (i.e., not parallel with
# other tests)
PY_TEST_EXCLUSIVE_LIST=""

# Append custom list of exclusive tests
if [[ ! -z "${TF_BUILD_EXTRA_EXCLUSIVE_INSTALL_TESTS}" ]]; then
  PY_TEST_EXCLUSIVE_LIST="${PY_TEST_EXCLUSIVE_LIST}:"\
"${TF_BUILD_EXTRA_EXCLUSIVE_INSTALL_TESTS}"
fi

# =============================================================================

echo "PY_TEST_WHITELIST: ${PY_TEST_WHITELIST}"
echo "PY_TEST_BLACKLIST: ${PY_TEST_BLACKLIST}"
echo "PY_TEST_GPU_BLACKLIST: ${PY_TEST_GPU_BLACKLIST}"


# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"


# Process input arguments
IS_VIRTUALENV=0
IS_GPU=0
while true; do
  if [[ "$1" == "--virtualenv" ]]; then
    IS_VIRTUALENV=1
  elif [[ "$1" == "--gpu" ]]; then
    IS_GPU=1
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

# Obtain the path to Python binary
if [[ ${IS_VIRTUALENV} == "1" ]]; then
  PYTHON_BIN_PATH="$(which python)"
else
  source tools/python_bin_path.sh
  # Assume: PYTHON_BIN_PATH is exported by the script above
fi

# Obtain the path to head/ghead binary (for log file printing)
HEAD_BIN="ghead"
if [[ -z $(which "${HEAD_BIN}") ]]; then
  # This is not Mac (which uses coreutils/ghead), use head.
  HEAD_BIN="head"
  if [[ -z $(which "${HEAD_BIN}") ]]; then
     die "Unable to obtain path to head or ghead"
  fi
fi

if [[ -z "${PYTHON_BIN_PATH}" ]]; then
  die "PYTHON_BIN_PATH was not provided. If this is not virtualenv, "\
"did you run configure?"
fi

# Append GPU-only test blacklist
if [[ ${IS_GPU} == "1" ]]; then
  PY_TEST_BLACKLIST="${PY_TEST_BLACKLIST}:${PY_TEST_GPU_BLACKLIST}"
fi

# Determine the major and minor versions of Python being used (e.g., 2.7)
# This info will be useful for determining the directory of the local pip
# installation of Python
PY_MAJOR_MINOR_VER=$(${PYTHON_BIN_PATH} -V 2>&1 | awk '{print $NF}' | cut -d. -f-2)

echo "Python binary path to be used in PIP install-test: ${PYTHON_BIN_PATH} "\
"(Major.Minor version: ${PY_MAJOR_MINOR_VER})"

# Avoid permission issues outside container
umask 000

# Directory from which the unit-test files will be run
PY_TEST_DIR_REL="pip_test/tests"
PY_TEST_DIR=$(realpath ${PY_TEST_DIR_REL})  # Get absolute path
rm -rf ${PY_TEST_DIR} && mkdir -p ${PY_TEST_DIR}

# Create test log directory
PY_TEST_LOG_DIR_REL=${PY_TEST_DIR_REL}/logs
PY_TEST_LOG_DIR=$(realpath ${PY_TEST_LOG_DIR_REL})  # Absolute path

mkdir ${PY_TEST_LOG_DIR}

# Copy source files that are required by the tests but are not included in the
# PIP package

# Look for local Python library directory
# pushd/popd avoids importing TensorFlow from the source directory.
pushd /tmp > /dev/null
TF_INSTALL_PATH=$(dirname \
    $("${PYTHON_BIN_PATH}" -c "import tensorflow as tf; print(tf.__file__)"))
popd > /dev/null

if [[ -z ${TF_INSTALL_PATH} ]]; then
  die "Failed to find path where TensorFlow is installed."
else
  echo "Found TensorFlow install path: ${TF_INSTALL_PATH}"
fi

echo "Copying some source directories required by Python unit tests but "\
"not included in install to TensorFlow install path: ${TF_INSTALL_PATH}"

# Files for tensorflow.python.tools
rm -rf ${TF_INSTALL_PATH}/python/tools
cp -r tensorflow/python/tools \
      ${TF_INSTALL_PATH}/python/tools
touch ${TF_INSTALL_PATH}/python/tools/__init__.py  # Make module visible

# Files for tensorflow.examples
rm -rf ${TF_INSTALL_PATH}/examples/image_retraining
mkdir -p ${TF_INSTALL_PATH}/examples/image_retraining
cp -r tensorflow/examples/image_retraining/retrain.py \
      ${TF_INSTALL_PATH}/examples/image_retraining/retrain.py
touch ${TF_INSTALL_PATH}/examples/__init__.py
touch ${TF_INSTALL_PATH}/examples/image_retraining/__init__.py

echo "Copying additional files required by tests to working directory "\
"for test: ${PY_TEST_DIR}"

# Image files required by some tests, e.g., images_ops_test.py

mkdir -p ${PY_TEST_DIR}/tensorflow/core/lib
rm -rf ${PY_TEST_DIR}/tensorflow/core/lib/jpeg
cp -r tensorflow/core/lib/jpeg ${PY_TEST_DIR}/tensorflow/core/lib
rm -rf ${PY_TEST_DIR}/tensorflow/core/lib/png
cp -r tensorflow/core/lib/png ${PY_TEST_DIR}/tensorflow/core/lib
rm -rf ${PY_TEST_DIR}/tensorflow/core/lib/gif
cp -r tensorflow/core/lib/gif ${PY_TEST_DIR}/tensorflow/core/lib

# Copy test data from tensorflow/contrib/ffmpeg

mkdir -p ${PY_TEST_DIR}/tensorflow/contrib/ffmpeg
rm -rf ${PY_TEST_DIR}/tensorflow/contrib/ffmpeg/testdata
cp -r tensorflow/contrib/ffmpeg/testdata ${PY_TEST_DIR}

# Run tests
DIR0=$(pwd)
ALL_PY_TESTS_0=$(find tensorflow/{contrib,examples,models,python,tensorboard} \
    -type f \( -name "*_test.py" -o -name "test_*.py" \) | sort)


# Subroutine for filtering test file names by a bazel tag.
filter_tests_by_bazel_tag() {
  # Usage: filter_tests_by_bazel_tag (--inclusive | --exclusive)
  #            <BAZEL_TAG> <INPUT_TESTS>
  #
  #   E.g., filter_tests_by_bazel_tag --inclusive "local"
  #             "dir1/test1.py dir2/test2.py"
  #
  # Use the flag --inclusive so that only the tests that have the tag will be
  # included in the returned string.
  # Use the flag --exclusive so that the returned string will consist of only
  # the tests that do not have the tag.
  # INPUT_TESTS are the name of the input Python unit test files, seperated by
  # spaces.
  #
  # The output string (through stdout) is: OUTPUT_TESTS | DISCARDED_TESTS
  # That is: a list of tests that passed the filter, followed by " | ",
  # followed by a list of tests that are discarded

  FILTER_MODE=$1
  TAG=$2
  INPUT_TESTS=$3

  # Input sanity checks
  if [[ "${FILTER_MODE}" != "--inclusive" ]] &&
     [[ "${FILTER_MODE}" != "--exclusive" ]]; then
    echo "ERROR: Unrecognized filter mode: ${FILTER_MODE}"
    exit 1
  fi
  if [[ -z "${TAG}" ]]; then
    echo "ERROR: Bazal tag is not supplied"
    exit 1
  fi
  if [[ -z "${INPUT_TESTS}" ]]; then
    echo "ERROR: INPUT_TESTS is not supplied"
    exit 1
  fi

  # Check bazel on path
  if [[ -z $(which bazel) ]]; then
    echo "ERROR: bazel is not on path"
    exit 1
  fi

  # Get all bazel targets that have the specified tag
  BAZEL_TARGETS=\
$(bazel query "kind(py_test, attr(tags, "${TAG}", //tensorflow/...))" | sort)

  TARGET_ALIASES=":"
  for TARGET in ${BAZEL_TARGETS}; do
    # Transform, e.g., //tensorflow/python/kernel_tests:xent_op_test -->
    #                  python-xent_op_test
    # to be compared with the transformed strings from the Python unit test
    # file names.
    TARGET_1=$(echo "${TARGET}" | sed "s/:/ /g")
    TARGET_PATH_1=$(echo "${TARGET_1}" | sed "s/\/\// /g" | sed "s/\// /g" \
                    | awk '{print $2}')
    TARGET_BASE_NAME=$(echo "${TARGET_1}" | awk '{print $NF}')
    TARGET_ALIAS="${TARGET_PATH_1}-${TARGET_BASE_NAME}"

    TARGET_ALIASES="${TARGET_ALIASES}${TARGET_ALIAS}:"
  done
  TARGET_ALIASES="${TARGET_ALIASES}:"

  # Filter the list of tests obtained from listing files with the bazel query
  # results.
  TESTS_PASSED_FILTER=""
  TESTS_BLOCKED_BY_FILTER=""
  for PY_TEST in ${INPUT_TESTS}; do
    # Transform, e.g., tensorflow/python/kernel_tests/xent_op_test.py -->
    #                  python-xent_op_test
    PY_TEST_PATH_1=$(echo "${PY_TEST}" | sed "s/\// /g" | awk '{print $2}')
    PY_TEST_BASE_NAME=$(echo "${PY_TEST}" | sed "s/\// /g" \
                        | awk '{print $NF}' | sed "s/\.py//g")
    PY_TEST_ALIAS="${PY_TEST_PATH_1}-${PY_TEST_BASE_NAME}"

    TO_INCLUDE=0
    if [[ "${TARGET_ALIASES}" == *"${PY_TEST_ALIAS}"* ]] && \
       [[ "${FILTER_MODE}" == "--inclusive" ]]; then
      TO_INCLUDE=1
    elif [[ "${TARGET_ALIASES}" != *"${PY_TEST_ALIAS}"* ]] && \
         [[ "${FILTER_MODE}" == "--exclusive" ]]; then
      TO_INCLUDE=1
    fi

    if [[ ${TO_INCLUDE} == 1 ]]; then
      TESTS_PASSED_FILTER="${TESTS_PASSED_FILTER} ${PY_TEST}"
    else
      TESTS_BLOCKED_BY_FILTER="${TESTS_BLOCKED_BY_FILTER} ${PY_TEST}"
    fi
  done

  echo "${TESTS_PASSED_FILTER} | ${TESTS_BLOCKED_BY_FILTER}"
}


if [[ ${TF_BUILD_FILTER_INSTALL_TESTS_BY_TAG} != "" ]]; then
  # Iteratively apply the filter tags
  TAGS=(${TF_BUILD_FILTER_INSTALL_TESTS_BY_TAG//,/ })
  for TAG in ${TAGS[@]}; do
    if [[ ${TAG} == "-"* ]]; then
      MODE="--exclusive"
      TAG_1=$(echo ${TAG} | sed 's/-//')
    else
      MODE="--inclusive"
      TAG_1=${TAG}
    fi

    FILTER_OUTPUT=$(filter_tests_by_bazel_tag ${MODE} \
                   "${TAG_1}" "${ALL_PY_TESTS_0}")
    ALL_PY_TESTS_0=$(echo "${FILTER_OUTPUT}" | cut -d \| -f 1)
    DISCARDED_TESTS=$(echo "${FILTER_OUTPUT}" | cut -d \| -f 2)
    N_DISCARDED=$(echo "${DISCARDED_TESTS}" | wc -w)

    echo ""
    echo "Skipping ${N_DISCARDED} test(s) due to filter tag \"${TAG}\":"
    echo "${DISCARDED_TESTS}"
    echo ""
  done
fi

# Move the exclusive tests to the back of the list
EXCLUSIVE_LIST="$(echo "${PY_TEST_EXCLUSIVE_LIST}" | sed -e 's/:/ /g')"

ALL_PY_TESTS=""
for TEST in ${ALL_PY_TESTS_0}; do
  if [[ ! ${PY_TEST_EXCLUSIVE_LIST} == *"${TEST}"* ]]; then
    ALL_PY_TESTS="${ALL_PY_TESTS} ${TEST}"
  fi
done

# Number of parallel (non-exclusive) tests
N_PAR_TESTS=$(echo ${ALL_PY_TESTS} | wc -w)
echo "Number of non-exclusive tests: ${N_PAR_TESTS}"

for TEST in ${EXCLUSIVE_LIST}; do
  ALL_PY_TESTS="${ALL_PY_TESTS} ${TEST}"
done

PY_TEST_COUNT=$(echo ${ALL_PY_TESTS} | wc -w)

if [[ ${PY_TEST_COUNT} -eq 0 ]]; then
  die "ERROR: Cannot find any tensorflow Python unit tests to run on install"
fi

# Iterate through all the Python unit test files using the installation
TEST_COUNTER=0
PASS_COUNTER=0
FAIL_COUNTER=0
SKIP_COUNTER=0
FAILED_TESTS=""
FAILED_TEST_LOGS=""

N_JOBS=$(grep -c ^processor /proc/cpuinfo)
if [[ -z ${N_JOBS} ]]; then
  # Try the Mac way of getting number of CPUs
  N_JOBS=$(sysctl -n hw.ncpu)
fi

if [[ -z ${N_JOBS} ]]; then
  N_JOBS=8
  echo "Cannot determine the number of processors"
  echo "Using default concurrent job counter ${N_JOBS}"
fi

if [[ ! -z "${TF_BUILD_SERIAL_INSTALL_TESTS}" ]] &&
   [[ "${TF_BUILD_SERIAL_INSTALL_TESTS}" != "0" ]]; then
  N_JOBS=1
fi

echo "Running Python tests-on-install with ${N_JOBS} concurrent jobs..."

ALL_PY_TESTS=(${ALL_PY_TESTS})
while true; do
  TEST_LOGS=""
  TEST_INDICES=""
  TEST_FILE_PATHS=""
  TEST_BASENAMES=""

  ITER_COUNTER=0
  while true; do
    # Break if the end is reached
    if [[ "${TEST_COUNTER}" -ge "${PY_TEST_COUNT}" ]]; then
      break;
    fi

    # for TEST_FILE_PATH in ${ALL_PY_TESTS}; do
    TEST_FILE_PATH=${ALL_PY_TESTS[TEST_COUNTER]}

    ((TEST_COUNTER++))
    ((ITER_COUNTER++))

    # If PY_TEST_WHITELIST is not empty, only the white-listed tests will be run
    if [[ ! -z ${PY_TEST_WHITELIST} ]] && \
      [[ ! ${PY_TEST_WHITELIST} == *"${TEST_FILE_PATH}"* ]]; then
      ((SKIP_COUNTER++))
      echo "Non-whitelisted test SKIPPED: ${TEST_FILE_PATH}"

      continue
    fi

    # If the test is in the black list, skip it
    if [[ ${PY_TEST_BLACKLIST} == *"${TEST_FILE_PATH}"* ]]; then
      ((SKIP_COUNTER++))
      echo "Blacklisted test SKIPPED: ${TEST_FILE_PATH}"
      continue
    fi

    TEST_INDICES="${TEST_INDICES} ${TEST_COUNTER}"
    TEST_FILE_PATHS="${TEST_FILE_PATHS} ${TEST_FILE_PATH}"

    # Copy to a separate directory to guard against the possibility of picking
    # up modules in the source directory
    cp ${TEST_FILE_PATH} ${PY_TEST_DIR}/

    TEST_BASENAME=$(basename "${TEST_FILE_PATH}")
    TEST_BASENAMES="${TEST_BASENAMES} ${TEST_BASENAME}"

    # Relative path of the test log. Use long path in case there are duplicate
    # file names in the Python tests
    TEST_LOG_REL="${PY_TEST_LOG_DIR_REL}/${TEST_FILE_PATH}.log"
    mkdir -p $(dirname ${TEST_LOG_REL})  # Create directory for log

    TEST_LOG=$(realpath ${TEST_LOG_REL})  # Absolute path
    TEST_LOGS="${TEST_LOGS} ${TEST_LOG}"

    # Launch test asynchronously
    "${SCRIPT_DIR}/py_test_delegate.sh" \
      "${PYTHON_BIN_PATH}" "${PY_TEST_DIR}/${TEST_BASENAME}" "${TEST_LOG}" &

    if [[ "${TEST_COUNTER}" -ge "${N_PAR_TESTS}" ]]; then
      # Run in exclusive mode
      if [[ "${TEST_COUNTER}" -gt "${N_PAR_TESTS}" ]]; then
        echo "Run test exclusively: ${PY_TEST_DIR}/${TEST_BASENAME}"
      fi
      break
    fi

    if [[ "${ITER_COUNTER}" -ge "${N_JOBS}" ]] ||
       [[ "${TEST_COUNTER}" -ge "${PY_TEST_COUNT}" ]]; then
      break
    fi

  done

  # Wait for all processes to complete
  wait

  TEST_LOGS=(${TEST_LOGS})
  TEST_FILE_PATHS=(${TEST_FILE_PATHS})
  TEST_BASENAMES=(${TEST_BASENAMES})

  K=0
  for TEST_INDEX in ${TEST_INDICES}; do
    TEST_FILE_PATH=${TEST_FILE_PATHS[K]}
    TEST_RESULT=$(tail -1 "${TEST_LOGS[K]}" | awk '{print $1}')
    ELAPSED_TIME=$(tail -1 "${TEST_LOGS[K]}" | cut -d' ' -f2-)

    PROG_STR="(${TEST_INDEX} / ${PY_TEST_COUNT})"
    # Check for pass or failure status of the test outtput and exit
    if [[ ${TEST_RESULT} -eq 0 ]]; then
      ((PASS_COUNTER++))

      echo "${PROG_STR} Python test-on-install PASSED (${ELAPSED_TIME}): ${TEST_FILE_PATH}"
    else
      ((FAIL_COUNTER++))

      FAILED_TESTS="${FAILED_TESTS} ${TEST_FILE_PATH}"
      FAILED_TEST_LOGS="${FAILED_TEST_LOGS} ${TEST_LOGS[K]}"

      echo "${PROG_STR} Python test-on-install FAILED (${ELAPSED_TIME}): ${TEST_FILE_PATH}"

      echo "  Log @: ${TEST_LOGS[K]}"
      echo "============== BEGINS failure log content =============="
      "${HEAD_BIN}" --lines=-1 "${TEST_LOGS[K]}"
      echo "============== ENDS failure log content =============="
      echo ""
    fi
    cd ${DIR0}

    # Clean up files for this test
    rm -f ${TEST_BASENAMES[K]}

    ((K++))
  done

  # Stop if the end is reached
  if [[ "${TEST_COUNTER}" -ge "${PY_TEST_COUNT}" ]]; then
    break;
  fi
done

# Clean up files copied for Python unit tests:
rm -rf ${TF_INSTALL_PATH}/python/tools
rm -rf ${TF_INSTALL_PATH}/examples/image_retraining
rm -rf ${PY_TEST_DIR}/tensorflow/core/lib/jpeg
rm -rf ${PY_TEST_DIR}/tensorflow/core/lib/png
rm -rf ${PY_TEST_DIR}/testdata

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
