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
#   TF_BUILD_INTEG_TEST_DENYLIST: Force skipping of specified integration tests
#       listed in INTEG_TESTS below.
#

# List of all integration tests to run, separated by spaces
INTEG_TESTS="ffmpeg_lib"

if [[ -z "${TF_BUILD_INTEG_TEST_DENYLIST}" ]]; then
  TF_BUILD_INTEG_TEST_DENYLIST=""
fi
echo ""
echo "=== Integration Tests ==="
echo "TF_BUILD_INTEG_TEST_DENYLIST = \"${TF_BUILD_INTEG_TEST_DENYLIST}\""

# Timeout (in seconds) for each integration test
TIMEOUT=1800

INTEG_TEST_ROOT="$(mktemp -d)"
LOGS_DIR=pip_test/integration_tests/logs

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
source "${SCRIPT_DIR}/builds_common.sh"

# Helper functions
cleanup() {
  rm -rf $INTEG_TEST_ROOT
}


die() {
  echo $@
  cleanup
  exit 1
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
    bazel test tensorflow/contrib/ffmpeg/default:ffmpeg_lib_uninstalled_test
    return $?
  else
    bazel test tensorflow/contrib/ffmpeg/default:ffmpeg_lib_installed_test \
        tensorflow/contrib/ffmpeg:decode_audio_op_test \
        tensorflow/contrib/ffmpeg:encode_audio_op_test
    return $?
  fi
}


# Run the integration tests
test_runner "integration test-on-install" \
    "${INTEG_TESTS}" "${TF_BUILD_INTEG_TEST_DENYLIST}" "${LOGS_DIR}"
