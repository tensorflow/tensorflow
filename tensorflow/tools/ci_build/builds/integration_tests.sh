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

# This script runs integration tests on the TensorFlow source code
# using a pip installation.
#
# Usage: integration_tests.sh [--virtualenv]
#
# If the flag --virtualenv is set, the script will use "python" as the Python
# binary path. Otherwise, it will use tools/python_bin_path.sh to determine
# the Python binary path.
#
# This script obeys the following environment variables (if exists):
#   TF_BUILD_INTEG_TEST_BLACKLIST: Force skipping of specified integration tests
#       listed in INTEG_TESTS below.
#

# List of all integration tests to run, separated by spaces
INTEG_TESTS="ffmpeg_lib"

if [[ -z "${TF_BUILD_INTEG_TEST_BLACKLIST}" ]]; then
  TF_BUILD_INTEG_TEST_BLACKLIST=""
fi
echo ""
echo "=== Integration Tests ==="
echo "TF_BUILD_INTEG_TEST_BLACKLIST = \"${TF_BUILD_INTEG_TEST_BLACKLIST}\""

# Timeout (in seconds) for each integration test
TIMEOUT=1800

INTEG_TEST_ROOT="$(mktemp -d)"
LOGS_DIR=pip_test/integration_tests/logs

# Helper functions
cleanup() {
  rm -rf $INTEG_TEST_ROOT
}


die() {
  echo $@
  cleanup
  exit 1
}


realpath() {
  [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}


# TODO(cais): This is similar to code in both test_tutorials.sh and
# test_installation.sh. It should be pulled into a common library.
run_in_directory() {
  DEST_DIR="$1"
  LOG_FILE="$2"
  INTEG_SCRIPT="$3"
  shift 3
  SCRIPT_ARGS=("$@")

  # Get the absolute path of the log file
  LOG_FILE_ABS=$(realpath "${LOG_FILE}")

  cp "${INTEG_SCRIPT}" "${DEST_DIR}"/
  SCRIPT_BASENAME=$(basename "${INTEG_SCRIPT}")

  if [[ ! -f "${DEST_DIR}/${SCRIPT_BASENAME}" ]]; then
    echo "FAILED to copy script ${INTEG_SCRIPT} to temporary directory "\
"${DEST_DIR}"
    return 1
  fi

  pushd "${DEST_DIR}" > /dev/null

  "${TIMEOUT_BIN}" --preserve-status ${TIMEOUT} \
    "${PYTHON_BIN_PATH}" "${SCRIPT_BASENAME}" ${SCRIPT_ARGS[@]} 2>&1 \
    > "${LOG_FILE_ABS}"

  rm -f "${SCRIPT_BASENAME}"
  popd > /dev/null

  if [[ $? != 0 ]]; then
    echo "Integration test \"${SCRIPT_BASENAME}\" FAILED"
    return 1
  fi

  return 0
}

# Determine the binary path for "timeout"
TIMEOUT_BIN="timeout"
if [[ -z "$(which ${TIMEOUT_BIN})" ]]; then
  TIMEOUT_BIN="gtimeout"
  if [[ -z "$(which ${TIMEOUT_BIN})" ]]; then
    die "Unable to locate binary path for timeout command"
  fi
fi
echo "Binary path for timeout: \"$(which ${TIMEOUT_BIN})\""

# Avoid permission issues outside Docker containers
umask 000

mkdir -p "${LOGS_DIR}" || die "Failed to create logs directory"
mkdir -p "${INTEG_TEST_ROOT}" || die "Failed to create test directory"

if [[ "$1" == "--virtualenv" ]]; then
  PYTHON_BIN_PATH="$(which python)"
else
  source tools/python_bin_path.sh
fi

if [[ -z "${PYTHON_BIN_PATH}" ]]; then
  die "PYTHON_BIN_PATH was not provided. If this is not virtualenv, "\
"did you run configure?"
else
  echo "Binary path for python: \"$PYTHON_BIN_PATH\""
fi

# Determine the TensorFlow installation path
# pushd/popd avoids importing TensorFlow from the source directory.
pushd /tmp > /dev/null
TF_INSTALL_PATH=$(dirname \
    $("${PYTHON_BIN_PATH}" -c "import tensorflow as tf; print(tf.__file__)"))
popd > /dev/null

echo "Detected TensorFlow installation path: ${TF_INSTALL_PATH}"

TEST_DIR="pip_test/integration"
mkdir -p "${TEST_DIR}" || \
    die "Failed to create test directory: ${TEST_DIR}"

# -----------------------------------------------------------
# ffmpeg_lib_test
test_ffmpeg_lib() {
  # If FFmpeg is not installed then run a test that assumes it is not installed.
  if [[ -z "$(which ffmpeg)" ]]; then
    bazel test tensorflow/contrib/ffmpeg/kernels:ffmpeg_lib_uninstalled_test
    return $?
  else
    bazel test tensorflow/contrib/ffmpeg/kernels:ffmpeg_lib_installed_test
    return $?
  fi
}


# Run the integration tests
# TODO(cais): This is similar to code in both test_tutorials.sh and
# test_installation.sh. It should be pulled into a common library.
NUM_INTEG_TESTS=$(echo "${INTEG_TESTS}" | wc -w)
INTEG_TESTS=(${INTEG_TESTS})

COUNTER=0
PASSED_COUNTER=0
FAILED_COUNTER=0
FAILED_TESTS=""
FAILED_TEST_LOGS=""
SKIPPED_COUNTER=0
for INTEG_TEST in ${INTEG_TESTS[@]}; do
  ((COUNTER++))
  STAT_STR="(${COUNTER} / ${NUM_INTEG_TESTS})"

  if [[ "${TF_BUILD_INTEG_TEST_BLACKLIST}" == *"${INTEG_TEST}"* ]]; then
    ((SKIPPED_COUNTER++))
    echo "${STAT_STR} Blacklisted integration test SKIPPED: ${INTEG_TEST}"
    continue
  fi

  START_TIME=$(date +'%s')

  LOG_FILE="${LOGS_DIR}/${INTEG_TEST}.log"
  rm -rf ${LOG_FILE} ||
  die "Unable to remove existing log file: ${LOG_FILE}"

  "test_${INTEG_TEST}" "${LOG_FILE}"
  TEST_RESULT=$?

  END_TIME=$(date +'%s')
  ELAPSED_TIME="$((${END_TIME} - ${START_TIME})) s"

  if [[ ${TEST_RESULT} == 0 ]]; then
    ((PASSED_COUNTER++))
    echo "${STAT_STR} Integration test PASSED: ${INTEG_TEST} "\
"(Elapsed time: ${ELAPSED_TIME})"
  else
    ((FAILED_COUNTER++))
    FAILED_TESTS="${FAILED_TESTS} ${INTEG_TEST}"
    FAILED_TEST_LOGS="${FAILED_TEST_LOGS} ${LOG_FILE}"

    echo "${STAT_STR} Integration test FAILED: ${INTEG_TEST} "\
"(Elapsed time: ${ELAPSED_TIME})"

    echo "============== BEGINS failure log content =============="
    cat ${LOG_FILE}
    echo "============== ENDS failure log content =============="
    echo ""
  fi
done

echo "${NUM_INTEG_TESTS} integration test(s): "\
"${PASSED_COUNTER} passed; ${FAILED_COUNTER} failed; ${SKIPPED_COUNTER} skipped"

if [[ ${FAILED_COUNTER} -eq 0  ]]; then
  echo ""
  echo "Integration tests SUCCEEDED"

  cleanup
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
  die "Integration tests FAILED"
fi
