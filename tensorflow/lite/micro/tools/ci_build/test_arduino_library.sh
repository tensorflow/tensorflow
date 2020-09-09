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

rm -rf ${TEMP_BUILD_DIR}

mkdir -p "${ARDUINO_HOME_DIR}/libraries"
mkdir -p ${TEMP_BUILD_DIR}

unzip -q ${LIBRARY_ZIP} -d "${ARDUINO_LIBRARIES_DIR}"

# Installs all dependencies for Arduino
InstallLibraryDependencies () {
  # Required by magic_wand
  ${ARDUINO_CLI_TOOL} lib install Arduino_LSM9DS1@1.1.0

  # Required by person_detection
  ${ARDUINO_CLI_TOOL} lib install JPEGDecoder@1.8.0
  # Patch to ensure works with nano33ble. This hack (deleting the entire
  # contents of the file) works with 1.8.0. If we bump the version, may need a
  # different patch.
  > ${ARDUINO_LIBRARIES_DIR}/JPEGDecoder/src/User_Config.h

  # Arducam, not available through Arduino library manager. This specific
  # commit is tested to work; if we bump the commit, we need to ensure that
  # the defines in ArduCAM/memorysaver.h are correct.
  wget -O /tmp/arducam-master.zip https://github.com/ArduCAM/Arduino/archive/e216049ba304048ec9bb29adfc2cc24c16f589b1/master.zip
  unzip /tmp/arducam-master.zip -d /tmp
  cp -r /tmp/Arduino-e216049ba304048ec9bb29adfc2cc24c16f589b1/ArduCAM "${ARDUINO_LIBRARIES_DIR}"
}

InstallLibraryDependencies

for f in ${ARDUINO_LIBRARIES_DIR}/tensorflow_lite/examples/*/*.ino; do
  ${ARDUINO_CLI_TOOL} compile --build-cache-path ${TEMP_BUILD_DIR} --build-path ${TEMP_BUILD_DIR} --fqbn arduino:mbed:nano33ble $f
done

rm -rf ${ARDUINO_LIBRARIES_DIR}
rm -rf ${TEMP_BUILD_DIR}
