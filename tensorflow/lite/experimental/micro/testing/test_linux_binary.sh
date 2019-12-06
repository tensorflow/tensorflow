#!/bin/bash -e
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# Tests a Linux binary by parsing the log output.
#
# First argument is the binary location.
# Second argument is a regular expression that's required to be in the output logs
# for the test to pass.

declare -r ROOT_DIR=`pwd`
declare -r TEST_TMPDIR=/tmp/test_linux_binary/
declare -r MICRO_LOG_PATH=${TEST_TMPDIR}/$1
declare -r MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt
mkdir -p ${MICRO_LOG_PATH}

ERROR_MSG="$1: FAIL - '$2' not found in logs."
print_error_and_exit() {
  echo ${ERROR_MSG}
  cat ${MICRO_LOG_FILENAME}
  exit 1
}

# This traps the signal from the test binary ($1) and checks if there was a
# segfault and adds that to the error log (which would otherwise be missing).
trap 'if [[ $? -eq 139 ]]; then echo "Segmentation fault" >> ${MICRO_LOG_FILENAME}; print_error_and_exit; fi' CHLD

# This trap statement prevents the bash script from segfaulting with a cryptic
# message like:
# tensorflow/lite/experimental/micro/testing/test_linux_binary.sh: line 44: 210514 Segmentation fault      $1 > ${MICRO_LOG_FILENAME} 2>&1
# What we get instead is purely another Segmentation fault text in the output.
trap '' SEGV

$1 > ${MICRO_LOG_FILENAME} 2>&1

if grep -q "$2" ${MICRO_LOG_FILENAME}
then
  echo "$1: PASS"
  exit 0
else
  print_error_and_exit
fi

