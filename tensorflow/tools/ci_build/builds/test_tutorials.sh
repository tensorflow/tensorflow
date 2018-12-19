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

# This script tests the tutorials and examples in TensorFlow source code
# on pip installation. The tutorial scripts are copied from source to a
# separate directory for testing. The script performs some very basic quality
# checks on the results, such as thresholding final accuracy, verifying
# decrement of loss with training, and verifying the existence of saved
# checkpoints and summaries files.
#
# Usage: test_tutorials.sh [--virtualenv]
#
# If the flag --virtualenv is set, the script will use "python" as the Python
# binary path. Otherwise, it will use tools/python_bin_path.sh to determine
# the Python binary path.
#
# This script obeys the following environment variables (if exists):
#   TUT_TESTS_BLACKLIST: Force skipping of specified tutorial tests listed
#                        in TUT_TESTS below.
#

# List of all tutorial tests to run, separated by spaces
TUT_TESTS="mnist_softmax mnist_with_summaries word2vec estimator_abalone"

if [[ -z "${TUT_TESTS_BLACKLIST}" ]]; then
  TF_BUILD_TUT_TEST_BLACKLIST=""
fi
echo ""
echo "=== Testing tutorials ==="
echo "TF_BUILD_TUT_TEST_BLACKLIST = \"${TF_BUILD_TUT_TEST_BLACKLIST}\""

# Timeout (in seconds) for each tutorial test
TIMEOUT=1800

TUT_TEST_ROOT=/tmp/tf_tutorial_test
TUT_TEST_DATA_DIR=/tmp/tf_tutorial_test_data
LOGS_DIR=pip_test/tutorial_tests/logs

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
source "${SCRIPT_DIR}/builds_common.sh"


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
mkdir -p "${TUT_TEST_ROOT}" || die "Failed to create test directory"

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

TEST_DIR="pip_test/tutorials"
mkdir -p "${TEST_DIR}" || \
    die "Failed to create test directory: ${TEST_DIR}"

# Copy folders required by mnist tutorials
mkdir -p "${TF_INSTALL_PATH}/examples/tutorials"
cp tensorflow/examples/tutorials/__init__.py \
    "${TF_INSTALL_PATH}/examples/tutorials/"
cp -r tensorflow/examples/tutorials/mnist \
    "${TF_INSTALL_PATH}/examples/tutorials/"

if [[ ! -d "${TF_INSTALL_PATH}/examples/tutorials/mnist" ]]; then
  die "FAILED: Unable to copy directory required by MNIST tutorials: "\
"${TF_INSTALL_PATH}/examples/tutorials/mnist"
fi


# -----------------------------------------------------------
# mnist_softmax
test_mnist_softmax() {
  LOG_FILE=$1

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    tensorflow/examples/tutorials/mnist/mnist_softmax.py \
    --data_dir="${TUT_TEST_DATA_DIR}/mnist"

  # Check final accuracy
  FINAL_ACCURACY=$(tail -1 "${LOG_FILE}")
  if [[ $(python -c "print(${FINAL_ACCURACY}>0.85)") != "True" ]] ||
     [[ $(python -c "print(${FINAL_ACCURACY}<=1.00)") != "True" ]]; then
    echo "mnist_softmax accuracy check FAILED: "\
"FINAL_ACCURACY = ${FINAL_ACCURACY}"
    return 1
  fi
}


# -----------------------------------------------------------
# mnist_with_summaries
test_mnist_with_summaries() {
  LOG_FILE=$1

  SUMMARIES_DIR="${TUT_TEST_ROOT}/summaries/mnist"
  rm -rf "${SUMMARIES_DIR}"
  # rm -rf "${TEST_DIR}/tensorflow"

  if [[ $? != 0 ]]; then
    echo "FAILED: unable to remove existing summaries directory: "\
"${SUMMARIES_DIR}"
    return 1
  fi

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    tensorflow/examples/tutorials/mnist/mnist_with_summaries.py \
    --data_dir="${TUT_TEST_DATA_DIR}/mnist" --log_dir="${SUMMARIES_DIR}"

  # Verify final accuracy
  FINAL_ACCURACY=$(grep "Accuracy at step" "${LOG_FILE}" \
                   | tail -1 | awk '{print $NF}')
  if [[ $(python -c "print(${FINAL_ACCURACY}>0.85)") != "True" ]] ||
     [[ $(python -c "print(${FINAL_ACCURACY}<=1.00)") != "True" ]]; then
    echo "mnist_with_summaries accuracy check FAILED: final accuracy = "\
"${FINAL_ACCURACY}"
    return 1
  fi

  # Verify that the summaries file have been generated
  if [[ ! -d "${SUMMARIES_DIR}" ]] ||
     [[ -z $(ls "${SUMMARIES_DIR}") ]]; then
    echo "FAILED: It appears that no summaries files were generated in "\
"${SUMMARIES_DIR}"
    return 1
  fi
}


# -----------------------------------------------------------
# cifar10_train
test_cifar10_train() {
  LOG_FILE=$1

  rm -rf "${TUT_TEST_ROOT}/cifar10_train"
  if [[ $? != 0 ]]; then
    echo "Unable to remove old cifar10_train directory"
    return 1
  fi

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    ${TF_MODELS_DIR}/tutorials/image/cifar10/cifar10_train.py \
    --data_dir="${TUT_TEST_DATA_DIR}/cifar10" --max_steps=50 \
    --train_dir="${TUT_TEST_ROOT}/cifar10_train"

  # Verify that final loss is less than initial loss
  INIT_LOSS=$(grep -o "loss = [0-9\.]*" "${LOG_FILE}" | head -1 | \
    awk '{print $NF}')
  FINAL_LOSS=$(grep -o "loss = [0-9\.]*" "${LOG_FILE}" | tail -1 | \
    awk '{print $NF}')

  if [[ $(python -c "print(${FINAL_LOSS}<${INIT_LOSS})") != "True" ]] ||
     [[ $(python -c "print(${INIT_LOSS}>=0)") != "True" ]] ||
     [[ $(python -c "print(${FINAL_LOSS}>=0)") != "True" ]]; then
    echo "cifar10_train loss check FAILED: "\
"FINAL_LOSS = ${FINAL_LOSS}; INIT_LOSS = ${INIT_LOSS}"
    return 1
  fi

  return 0
}


# -----------------------------------------------------------
# word2vec_test
test_word2vec() {
  LOG_FILE=$1

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    tensorflow/examples/tutorials/word2vec/word2vec_basic.py
}


# -----------------------------------------------------------
# ptb_word_lm
test_ptb_word_lm() {
  LOG_FILE=$1

  PTB_DATA_URL="http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"

  DATA_DIR="${TUT_TEST_DATA_DIR}/ptb"
  if [[ ! -f "${DATA_DIR}/simple-examples/data/ptb.train.txt" ]] || \
     [[ ! -f "${DATA_DIR}/simple-examples/data/ptb.valid.txt" ]] || \
     [[ ! -f "${DATA_DIR}/simple-examples/data/ptb.test.txt" ]]; then
    # Download and extract data
    echo "Downloading and extracting PTB data from \"${PTB_DATA_URL}\" to "\
"${DATA_DIR}"

    mkdir -p ${DATA_DIR}
    pushd ${DATA_DIR} > /dev/null
    curl --retry 5 --retry-delay 10 -O "${PTB_DATA_URL}" || \
        die "Failed to download data file for ptb_world_lm tutorial from "\
"${PTB_DATA_URL}"
    tar -xzf $(basename "${PTB_DATA_URL}")
    rm -f $(basename "${PTB_DATA_URL}")
    popd > /dev/null

    if [[ ! -d "${DATA_DIR}/simple-examples/data" ]]; then
      echo "FAILED to download and extract data for \"ptb_word_lm\""
      return 1
    fi
  fi

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    "${TF_MODELS_DIR}/tutorials/rnn/ptb/ptb_word_lm.py" \
    --data_path="${DATA_DIR}/simple-examples/data" --model test

  if [[ $? != 0 ]]; then
    echo "Tutorial test \"ptb_word_lm\" FAILED"
    return 1
  fi

  # Extract epoch initial and final training perplexities
  INIT_PERPL=$(grep -o "[0-9\.]* perplexity: [0-9\.]*" "${LOG_FILE}" | head -1 | awk '{print $NF}')
  FINAL_PERPL=$(grep -o "[0-9\.]* perplexity: [0-9\.]*" "${LOG_FILE}" | tail -1 | awk '{print $NF}')

  echo "INIT_PERPL=${INIT_PERPL}"
  echo "FINAL_PERPL=${FINAL_PERPL}"

  if [[ $(python -c "print(${FINAL_PERPL}<${INIT_PERPL})") != "True" ]] ||
     [[ $(python -c "print(${INIT_PERPL}>=0)") != "True" ]] ||
     [[ $(python -c "print(${FINAL_PERPL}>=0)") != "True" ]]; then
    echo "ptb_word_lm perplexity check FAILED: "\
"FINAL_PERPL = ${FINAL_PERPL}; INIT_PERPL = ${INIT_PERPL}"
    return 1
  fi
}

# Run the tutorial tests
test_runner "tutorial test-on-install" \
    "${TUT_TESTS}" "${TF_BUILD_TUT_TEST_BLACKLIST}" "${LOGS_DIR}"
