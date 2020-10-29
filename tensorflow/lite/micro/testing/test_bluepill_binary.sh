#!/bin/bash -e
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# Tests a 'bluepill' STM32F103 ELF by parsing the log output of Renode emulation.
#
# First argument is the ELF location.
# Second argument is a regular expression that's required to be in the output logs
# for the test to pass.
#
# This script must be run from the top-level folder of the tensorflow github
# repository as it mounts `pwd` to the renode docker image (via docker run -v)
# and paths in the docker run command assume the entire tensorflow repo is mounted.

declare -r ROOT_DIR=`pwd`
declare -r TEST_TMPDIR=/tmp/test_bluepill_binary/
declare -r MICRO_LOG_PATH=${TEST_TMPDIR}
declare -r MICRO_LOG_FILENAME=${MICRO_LOG_PATH}logs.txt
mkdir -p ${MICRO_LOG_PATH}

declare -r RENODE_TEST_SCRIPT=${ROOT_DIR}/tensorflow/lite/micro/tools/make/downloads/renode/test.sh
if [ ! -f "${RENODE_TEST_SCRIPT}" ]; then
  echo "The renode test script: ${RENODE_TEST_SCRIPT} does not exist. Please " \
       "make sure that you have correctly installed Renode for TFLM. See " \
       "tensorflow/lite/micro/docs/renode.md for more details."
  exit 1
fi

if ! ${RENODE_TEST_SCRIPT} &> /dev/null
then
  echo "The following command failed: ${RENODE_TEST_SCRIPT}. Please " \
       "make sure that you have correctly installed Renode for TFLM. See " \
       "tensorflow/lite/micro/docs/renode.md for more details."
  exit 1
fi


# This check ensures that we only have a single $MICRO_LOG_FILENAME. Without it,
# renode will do a log rotation and there will be multiple files such as
# $MICRO_LOG_FILENAME.1 $MICRO_LOG_FILENAME.2 etc.
if [ -e $MICRO_LOG_FILENAME ]; then
    rm $MICRO_LOG_FILENAME &> /dev/null
fi;

exit_code=0

if ! BIN=${ROOT_DIR}/$1 \
  SCRIPT=${ROOT_DIR}/tensorflow/lite/micro/testing/bluepill.resc \
  LOGFILE=$MICRO_LOG_FILENAME \
  EXPECTED="$2" \
  ${RENODE_TEST_SCRIPT} \
  ${ROOT_DIR}/tensorflow/lite/micro/testing/bluepill.robot \
  -r $TEST_TMPDIR &> ${MICRO_LOG_PATH}robot_logs.txt
then
  exit_code=1
fi

echo "LOGS:"
# Extract output from renode log
cat ${MICRO_LOG_FILENAME} |grep 'uartSemihosting' |sed 's/^.*from start] *//g'
if [ $exit_code -eq 0 ]
then
  echo "$1: PASS"
else
  echo "$1: FAIL - '$2' not found in logs."
fi
exit $exit_code
