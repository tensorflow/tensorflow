#!/bin/bash -e
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#
# Parameters:
#  ${1} - path to a binary to test or directory (all *_test will be run).
#  ${2} - String that is checked for pass/fail.
#  ${3} - target (e.g. cortex_m_generic.)

set -e

BINARY_TO_TEST=${1}
PASS_STRING=${2}
TARGET=${3}

RESULTS_DIRECTORY=/tmp/${TARGET}_logs
MICRO_LOG_FILENAME=${RESULTS_DIRECTORY}/logs.txt
mkdir -p ${RESULTS_DIRECTORY}

FVP="FVP_Corstone_SSE-300_Ethos-U55 "
FVP+="--cpulimit 1 "
FVP+="-C mps3_board.visualisation.disable-visualisation=1 "
FVP+="-C mps3_board.telnetterminal0.start_telnet=0 "
FVP+='-C mps3_board.uart0.out_file="-" '
FVP+='-C mps3_board.uart0.unbuffered_output=1'
${FVP} ${BINARY_TO_TEST} | tee ${MICRO_LOG_FILENAME}

if grep -q "$PASS_STRING" ${MICRO_LOG_FILENAME}
then
  echo "$BINARY_TO_TEST: PASS"
  exit 0
else
  echo "$BINARY_TO_TEST: FAIL - '$PASS_STRING' not found in logs."
  exit 1
fi
