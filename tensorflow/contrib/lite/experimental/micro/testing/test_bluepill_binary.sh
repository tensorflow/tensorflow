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
# Tests a 'bluepill' STM32F103 ELF by parsing the log output of Renode emulation.
#
# First argument is the ELF location.
# Second argument is a regular expression that's required to be in the output logs
# for the test to pass.

declare -r ROOT_DIR=`pwd`
declare -r TEST_TMPDIR=/tmp/test_bluepill_binary/
declare -r MICRO_LOG_PATH=${TEST_TMPDIR}
declare -r MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt
mkdir -p ${MICRO_LOG_PATH}

docker build -t renode_bluepill \
  -f ${ROOT_DIR}/tensorflow/contrib/lite/experimental/micro/testing/Dockerfile.bluepill \
  ${ROOT_DIR}/tensorflow/contrib/lite/experimental/micro/testing/

docker run \
  --log-driver=none -a stdout -a stderr \
  -v ${ROOT_DIR}:/workspace \
  -v /tmp:/tmp \
  -it renode_bluepill \
  /bin/bash -c "renode -P 5000 --disable-xwt -e '
\$bin?=@/workspace/$1
s @/workspace/tensorflow/contrib/lite/experimental/micro/testing/bluepill.resc
' 2>&1 >${MICRO_LOG_FILENAME}"

echo "LOGS:"
cat ${MICRO_LOG_FILENAME}

if grep -q "$2" ${MICRO_LOG_FILENAME}
then
  echo "$1: PASS"
  exit 0
else
  echo "$1: FAIL - '$2' not found in logs."
  exit 1
fi

