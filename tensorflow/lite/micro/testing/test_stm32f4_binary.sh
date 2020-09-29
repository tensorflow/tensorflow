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
# Tests a 'stm32f4' STM32F4 ELF by parsing the log output of Renode emulation.
#
# First argument is the ELF location.
# Second argument is a regular expression that's required to be in the output logs
# for the test to pass.
#
# This script must be run from the top-level folder of the tensorflow github
# repository as it mounts `pwd` to the renode docker image (via docker run -v)
# and paths in the docker run command assume the entire tensorflow repo is mounted.

declare -r ROOT_DIR=`pwd`
declare -r EXPECTED=${2}
declare -r TIMEOUT_SEC=30
declare -r SCRIPT=${ROOT_DIR}/tensorflow/lite/micro/testing/stm32f4.resc
declare -r ROBOT_SCRIPT=tensorflow/lite/micro/testing/stm32f4.robot
declare -r TEST_TMPDIR=/tmp/test_stm32f4_binary
declare -r MICRO_LOG_PATH=${TEST_TMPDIR}
declare -r MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt
declare -r RENODE_PATH=/opt/renode/tests
declare -r RENODE_KEYWORDS_FILENAME=renode-keywords.robot
declare -r BIN_TO_TEST=${ROOT_DIR}/${1}
declare -r RESULTS_DIR=`basename ${BIN_TO_TEST}`
mkdir -p ${MICRO_LOG_PATH}

renode_found=1
RENODE_TEST=`which renode-test` && : || renode_found=0
if [ $renode_found -eq 0 ]
then
    RENODE_TEST=${RENODE_PATH}/test.sh  # When built from source there is no renode-test.
    if [ ! -f "$RENODE_TEST" ]; then
        echo "Renode not installed, please install it. For example:"
        echo "mkdir <your/path>/renode"
        echo "curl -L https://github.com/renode/renode/releases/download/<version>/"\
"renode-<version>.linux-portable.tar.gz | tar -C <your/path>/renode --strip-components=1 -zx"
        echo "export PATH=\$PATH:<your/path/to/unpacked/renode/where/test.sh/reside>"
        echo "ln -s <your/path/to/unpacked/renode>/test.sh <your/path/to/unpacked/renode>/renode-test"
        exit 1
    fi
fi

# Resolve path, which may be different with the portable version.
RENODE_KEYWORDS=${RENODE_PATH}/${RENODE_KEYWORDS_FILENAME}
if [ ! -f "$RENODE_KEYWORDS" ]; then
    REAL_PATH=`dirname ${RENODE_TEST}`
    RENODE_KEYWORDS=`find ${REAL_PATH} -name ${RENODE_KEYWORDS_FILENAME}`
    if [ ! -f "$RENODE_KEYWORDS" ]; then
        echo "${RENODE_KEYWORDS_FILENAME} not found"
        exit 1
    fi
fi

exit_code=0
${RENODE_TEST} --variable RENODE_KEYWORDS:${RENODE_KEYWORDS} --variable BIN:${BIN_TO_TEST} \
                 --variable SCRIPT:${SCRIPT} --variable EXPECTED:"${EXPECTED}" --variable TIMEOUT:${TIMEOUT_SEC} \
                 -r ${RESULTS_DIR} ${ROBOT_SCRIPT} 2>&1 >${MICRO_LOG_FILENAME} && : || exit_code=1

cat ${MICRO_LOG_FILENAME}
if [ $exit_code -eq 0 ]
then
  echo "$1: PASS"
else
  echo "$1: FAIL - '$2' not found in logs."
fi
exit $exit_code
