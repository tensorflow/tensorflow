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
# Bash unit tests for TensorFlow Debugger (tfdbg) Python examples that do not
# involve downloading data. Also tests the binary offline_analyzer.
#
# Command-line flags:
#   --virtualenv: (optional) If set, will test the examples and binaries
#     against pip install of TensorFlow in a virtualenv.

set -e

IS_VIRTUALENV=0
PYTHON_BIN_PATH=""
while true; do
  if [[ -z "$1" ]]; then
    break
  elif [[ "$1" == "--virtualenv" ]]; then
    IS_VIRTUALENV=1
    PYTHON_BIN_PATH=$(which python)
    echo
    echo "IS_VIRTUALENV = ${IS_VIRTUALENV}"
    echo "PYTHON_BIN_PATH = ${PYTHON_BIN_PATH}"
    echo "Will test tfdbg examples and binaries against virtualenv pip install."
    echo
  fi
  shift 1
done

if [[ -z "${PYTHON_BIN_PATH}" ]]; then
  DEBUG_FIBONACCI_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_fibonacci"
  DEBUG_ERRORS_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_errors"
  DEBUG_MNIST_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_mnist"
  DEBUG_TFLEARN_IRIS_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_tflearn_iris"
  OFFLINE_ANALYZER_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/offline_analyzer"
else
  DEBUG_FIBONACCI_BIN="${PYTHON_BIN_PATH} -m tensorflow.python.debug.examples.debug_fibonacci"
  DEBUG_ERRORS_BIN="${PYTHON_BIN_PATH} -m tensorflow.python.debug.examples.debug_errors"
  DEBUG_MNIST_BIN="${PYTHON_BIN_PATH} -m tensorflow.python.debug.examples.debug_mnist"
  DEBUG_TFLEARN_IRIS_BIN="${PYTHON_BIN_PATH} -m tensorflow.python.debug.examples.debug_tflearn_iris"
  OFFLINE_ANALYZER_BIN="${PYTHON_BIN_PATH} -m tensorflow.python.debug.cli.offline_analyzer"
fi

# Override the default ui_type=curses to allow the test to pass in a tty-less
# test environment.
cat << EOF | ${DEBUG_FIBONACCI_BIN} --ui_type=readline
run
exit
EOF

cat << EOF | ${DEBUG_ERRORS_BIN} --error=no_error --ui_type=readline
run
exit
EOF

# Use a large enough "run -t" number to let the process end properly.
cat << EOF | ${DEBUG_MNIST_BIN} --debug --fake_data --ui_type=readline
run -f has_inf_or_nan
run -t 1000
EOF

cat << EOF | ${DEBUG_TFLEARN_IRIS_BIN} --debug --fake_data --train_steps=2 --ui_type=readline
run -f has_inf_or_nan
EOF

# Test offline_analyzer.
# TODO(cais): Generate an actual debug dump and load it with offline_analyzer,
# so that we can test the binary runs with a non-error exit code.
set +e
OUTPUT=$(${OFFLINE_ANALYZER_BIN} 2>&1)
set -e

EXPECTED_OUTPUT="ERROR: dump_dir flag is empty."
if [[ "${OUTPUT}" != "${EXPECTED_OUTPUT}" ]]; then
  echo "ERROR: offline_analyzer output didn't match expectation: ${OUTPUT}" 1>&2
  echo "Expected output: ${EXPECTED_OUTPUT}"
  exit 1
fi

echo
echo "SUCCESS: tfdbg examples and binaries test PASSED"
