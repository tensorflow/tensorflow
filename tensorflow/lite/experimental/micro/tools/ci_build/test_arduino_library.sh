#!/usr/bin/env bash
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
# Tests an individual Arduino library. Because libraries need to be installed
# globally, this can cause problems with previously-installed modules, so we
# recommend that you only run this within a VM.

set -e

ARDUINO_HOME_DIR=${HOME}/Arduino
ARDUINO_LIBRARIES_DIR=${ARDUINO_HOME_DIR}/libraries
ARDUINO_CLI_TOOL=/tmp/arduino-cli
# Necessary due to bug in arduino-cli that allows it to build files in pwd
TEMP_BUILD_DIR=/tmp/tflite-arduino-build

LIBRARY_ZIP=${1}

rm -rf ${ARDUINO_LIBRARIES_DIR}
rm -rf ${TEMP_BUILD_DIR}

mkdir -p ${ARDUINO_HOME_DIR}/libraries
mkdir -p ${TEMP_BUILD_DIR}

unzip -q ${LIBRARY_ZIP} -d ${ARDUINO_LIBRARIES_DIR}

# Change into this dir before running the tests
cd ${TEMP_BUILD_DIR}

for f in ${ARDUINO_LIBRARIES_DIR}/*/examples/*/*.ino; do
  ${ARDUINO_CLI_TOOL} compile --fqbn arduino:mbed:nano33ble $f
done
