#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# Filter out LOG(INFO)
export TF_CPP_MIN_LOG_LEVEL=1

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
  DEBUG_FIBONACCI_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_fibonacci_v2"
  DEBUG_MNIST_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_mnist_v2"
else
  DEBUG_FIBONACCI_BIN="${PYTHON_BIN_PATH} -m tensorflow.python.debug.examples.v2.debug_fibonacci"
  DEBUG_MNIST_BIN="${PYTHON_BIN_PATH} -m tensorflow.python.debug.examples.v2.debug_mnist"
fi

# Verify fibonacci runs normally without additional flags
${DEBUG_FIBONACCI_BIN} --tensor_size=2

# Verify mnist runs normally without additional flags
${DEBUG_MNIST_BIN} --max_steps=4 --fake_data

# Verify mnist does not break with check_numerics enabled on first iteration
# check_numerics should not cause non-zero exit code on a single train step
${DEBUG_MNIST_BIN} --max_steps=1 --fake_data --check_numerics

# Verify check_numerics exits with non-zero exit code
! ${DEBUG_MNIST_BIN} --max_steps=4 --fake_data --check_numerics

# Verify that dumping works properly.
TMP_DIR="$(mktemp -d)"
${DEBUG_MNIST_BIN} --max_steps=4 --fake_data --dump_dir="${TMP_DIR}"

# Check that the .execution and .graph_execution_traces are not empty,
# i.e., the content of the circular buffer have been written to the disk.
EXECUTION_FILE="$(ls ${TMP_DIR}/*.execution)"
EXECUTION_FILE_SIZE=$(du -b "${EXECUTION_FILE}" | awk '{print $1}')
echo "Size of ${EXECUTION_FILE}: ${EXECUTION_FILE_SIZE}"
if [[ "${EXECUTION_FILE_SIZE}" == "0" ]]; then
  echo "ERROR: ${EXECUTION_FILE} is unexpectedly empty."
  exit 1
fi

GRAPH_TRACES_FILE="$(ls ${TMP_DIR}/*.graph_execution_traces)"
GRAPH_TRACES_FILE_SIZE=$(du -b "${EXECUTION_FILE}" | awk '{print $1}')
echo "Size of ${GRAPH_TRACES_FILE}: ${GRAPH_TRACES_FILE_SIZE}"
if [[ "${GRAPH_TRACES_FILE_SIZE}" == "0" ]]; then
  echo "ERROR: ${GRAPH_TRACES_FILE} is unexpectedly empty."
  exit 1
fi

rm -rf "${TMP_DIR}"

echo "SUCCESS: tfdbg examples and binaries test PASSED"
