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
declare -r TEST_TMPDIR=/tmp/test_bluepill_binary/
declare -r MICRO_LOG_PATH=${TEST_TMPDIR}/$1
declare -r MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt
mkdir -p ${MICRO_LOG_PATH}

$1 2>&1 | tee ${MICRO_LOG_FILENAME}

if grep -q "$2" ${MICRO_LOG_FILENAME}
then
  echo "$1: PASS"
  exit 0
else
  echo "$1: FAIL - '$2' not found in logs."
  exit 1
fi

