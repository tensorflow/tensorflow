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
TUT_TESTS="mnist_softmax mnist_with_summaries cifar10_train "\
"word2vec_test word2vec_optimized_test ptb_word_lm translate_test"

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

# Helper functions
die() {
  echo $@
  exit 1
}


realpath() {
  [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}


run_in_directory() {
  DEST_DIR="$1"
  LOG_FILE="$2"
  TUT_SCRIPT="$3"
  shift 3
  SCRIPT_ARGS=("$@")

  # Get the absolute path of the log file
  LOG_FILE_ABS=$(realpath "${LOG_FILE}")

  cp "${TUT_SCRIPT}" "${DEST_DIR}"/
  SCRIPT_BASENAME=$(basename "${TUT_SCRIPT}")

  if [[ ! -f "${DEST_DIR}/${SCRIPT_BASENAME}" ]]; then
    echo "FAILED to copy script ${TUT_SCRIPT} to temporary directory "\
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
    echo "Tutorial test \"${SCRIPT_BASENAME}\" FAILED"
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
  if [[ $(echo "${FINAL_ACCURACY}>0.85" | bc -l) != "1" ]] ||
     [[ $(echo "${FINAL_ACCURACY}<=1.00" | bc -l) != "1" ]]; then
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
    --data_dir="${TUT_TEST_DATA_DIR}/mnist" --summaries_dir="${SUMMARIES_DIR}"

  # Verify final accuracy
  FINAL_ACCURACY=$(tail -1 "${LOG_FILE}" | awk '{print $NF}')
  if [[ $(echo "${FINAL_ACCURACY}>0.85" | bc -l) != "1" ]] ||
     [[ $(echo "${FINAL_ACCURACY}<=1.00" | bc -l) != "1" ]]; then
    echo "mnist_with_summaries accuracy check FAILED: ${FINAL_ACCURACY}<0.90"
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
    tensorflow/models/image/cifar10/cifar10_train.py \
    --data_dir="${TUT_TEST_DATA_DIR}/cifar10" --max_steps=50 \
    --train_dir="${TUT_TEST_ROOT}/cifar10_train"

  # Verify that final loss is less than initial loss
  INIT_LOSS=$(grep -o "loss = [0-9\.]*" "${LOG_FILE}" | head -1 | \
    awk '{print $NF}')
  FINAL_LOSS=$(grep -o "loss = [0-9\.]*" "${LOG_FILE}" | tail -1 | \
    awk '{print $NF}')

  if [[ $(echo "${FINAL_LOSS}<${INIT_LOSS}" | bc -l) != "1" ]] ||
     [[ $(echo "${INIT_LOSS}>=0" | bc -l) != "1" ]] ||
     [[ $(echo "${FINAL_LOSS}>=0" | bc -l) != "1" ]]; then
    echo "cifar10_train loss check FAILED: "\
"FINAL_LOSS = ${FINAL_LOSS}; INIT_LOSS = ${INIT_LOSS}"
    return 1
  fi

  # Check ckpt files
  if [[ ! -f "${TUT_TEST_ROOT}/cifar10_train/model.ckpt-0" ]] ||
    [[ ! -f "${TUT_TEST_ROOT}/cifar10_train/model.ckpt-49" ]]; then
    echo "FAILED: cifar10_train did not generate expected model checkpoint files"
    return 1
  fi

  return 0
}


# -----------------------------------------------------------
# word2vec_test
test_word2vec_test() {
  LOG_FILE=$1

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    tensorflow/models/embedding/word2vec_test.py
}


# -----------------------------------------------------------
# word2vec_optimized_test
test_word2vec_optimized_test() {
  LOG_FILE=$1

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    tensorflow/models/embedding/word2vec_optimized_test.py
}


# -----------------------------------------------------------
# ptb_word_lm
test_ptb_word_lm() {
  LOG_FILE=$1

  PTB_DATA_URL="http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"

  DATA_DIR="${TUT_TEST_DATA_DIR}/ptb"
  if [[ ! -d "${DATA_DIR}/simple-examples/data" ]]; then
    # Download and extract data
    echo "Downloading and extracting PTB data from \"${PTB_DATA_URL}\" to "\
"${DATA_DIR}"

    mkdir -p ${DATA_DIR}
    pushd ${DATA_DIR} > /dev/null
    curl -O "${PTB_DATA_URL}"
    tar -xzf $(basename "${PTB_DATA_URL}")
    rm -f $(basename "${PTB_DATA_URL}")
    popd > /dev/null

    if [[ ! -d "${DATA_DIR}/simple-examples/data" ]]; then
      echo "FAILED to download and extract data for \"ptb_word_lm\""
      return 1
    fi
  fi

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    tensorflow/models/rnn/ptb/ptb_word_lm.py \
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

  if [[ $(echo "${FINAL_PERPL}<${INIT_PERPL}" | bc -l) != "1" ]] ||
     [[ $(echo "${INIT_PERPL}>=0" | bc -l) != "1" ]] ||
     [[ $(echo "${FINAL_PERPL}>=0" | bc -l) != "1" ]]; then
    echo "ptb_word_lm perplexity check FAILED: "\
"FINAL_PERPL = ${FINAL_PERPL}; INIT_PERPL = ${INIT_PERPL}"
    return 1
  fi
}


# -----------------------------------------------------------
# translate_test
test_translate_test() {
  LOG_FILE=$1

  run_in_directory "${TEST_DIR}" "${LOG_FILE}" \
    tensorflow/models/rnn/translate/translate.py --self_test=True
}


# Run the tutorial tests
NUM_TUT_TESTS=$(echo "${TUT_TESTS}" | wc -w)
TUT_TESTS=(${TUT_TESTS})

COUNTER=0
PASSED_COUNTER=0
FAILED_COUNTER=0
FAILED_TESTS=""
FAILED_TEST_LOGS=""
SKIPPED_COUNTER=0
for TUT_TEST in ${TUT_TESTS[@]}; do
  ((COUNTER++))
  STAT_STR="(${COUNTER} / ${NUM_TUT_TESTS})"

  if [[ "${TF_BUILD_TUT_TEST_BLACKLIST}" == *"${TUT_TEST}"* ]]; then
    ((SKIPPED_COUNTER++))
    echo "${STAT_STR} Blacklisted tutorial test SKIPPED: ${TUT_TEST}"
    continue
  fi

  START_TIME=$(date +'%s')

  LOG_FILE="${LOGS_DIR}/${TUT_TEST}.log"
  rm -rf ${LOG_FILE} ||
  die "Unable to remove existing log file: ${LOG_FILE}"

  "test_${TUT_TEST}" "${LOG_FILE}"
  TEST_RESULT=$?

  END_TIME=$(date +'%s')
  ELAPSED_TIME="$((${END_TIME} - ${START_TIME})) s"

  if [[ ${TEST_RESULT} == 0 ]]; then
    ((PASSED_COUNTER++))
    echo "${STAT_STR} Tutorial test-on-install PASSED: ${TUT_TEST} "\
"(Elapsed time: ${ELAPSED_TIME})"
  else
    ((FAILED_COUNTER++))
    FAILED_TESTS="${FAILED_TESTS} ${TUT_TEST}"
    FAILED_TEST_LOGS="${FAILED_TEST_LOGS} ${LOG_FILE}"

    echo "${STAT_STR} Tutorial test-on-install FAILED: ${TUT_TEST} "\
"(Elapsed time: ${ELAPSED_TIME})"

    echo "============== BEGINS failure log content =============="
    cat ${LOG_FILE}
    echo "============== ENDS failure log content =============="
    echo ""
  fi
done

echo "${NUM_TUT_TESTS} tutorial test(s): "\
"${PASSED_COUNTER} passed; ${FAILED_COUNTER} failed; ${SKIPPED_COUNTER} skipped"

if [[ ${FAILED_COUNTER} -eq 0  ]]; then
  echo ""
  echo "Tutorial test-on-install SUCCEEDED"

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
  die "Tutorial test-on-install FAILED"
fi
