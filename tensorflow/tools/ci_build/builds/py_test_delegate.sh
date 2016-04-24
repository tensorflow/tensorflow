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
#
# This file runs a Python test file using specified Python binary, and store
# the test log, along with exit code and elapsed time in a log file.
#
# Usage: test_delegate.sh <PYTHON_BIN_PATH> <TEST_PATH> <TEST_LOG>

PYTHON_BIN_PATH=$1
TEST_PATH=$2
TEST_LOG=$3

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
source "${SCRIPT_DIR}/builds_common.sh"

rm -f ${TEST_LOG}

TEST_DIR=$(dirname ${TEST_PATH})
cd ${TEST_DIR}

START_TIME=$(date +'%s%N')

${PYTHON_BIN_PATH} ${TEST_PATH} >${TEST_LOG} 2>&1
TEST_EXIT_CODE=$?

END_TIME=$(date +'%s%N')

ELAPSED=$(calc_elapsed_time "${START_TIME}" "${END_TIME}")

echo "${TEST_EXIT_CODE} ${ELAPSED}" >> ${TEST_LOG}
