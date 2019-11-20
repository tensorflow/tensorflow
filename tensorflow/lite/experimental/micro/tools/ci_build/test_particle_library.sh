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
# Tests an individual Particle library. Because libraries need to be installed
# globally, this can cause problems with previously-installed modules, so we
# recommend that you only run this within a VM.

set -e

PARTICLE_HOME_DIR=${HOME}/Particle
PARTICLE_LIBRARIES_DIR=${PARTICLE_HOME_DIR}/libraries
PARTICLE_CLI_TOOL=/tmp/particle
TEMP_BUILD_DIR=/tmp/tflite-particle-build

LIBRARY_ZIP=${1}

rm -rf ${TEMP_BUILD_DIR}

mkdir -p ${PARTICLE_HOME_DIR}/libraries
mkdir -p ${TEMP_BUILD_DIR}

unzip -q ${LIBRARY_ZIP} -d ${PARTICLE_LIBRARIES_DIR}

# Installs all dependencies for Particle
InstallLibraryDependencies () {
    # Required by magic_wand
    ${PARTICLE_CLI_TOOL} library install --dest ${PARTICLE_LIBRARIES_DIR} LSM9DS1_FIFO
    ${PARTICLE_CLI_TOOL} library install --dest ${PARTICLE_LIBRARIES_DIR} Adafruit_Sensor
    
    # Required by micro_speech
    ${PARTICLE_CLI_TOOL} library install --dest ${PARTICLE_LIBRARIES_DIR} ADCDMAGen3_RK
}

InstallLibraryDependencies

# Change into this dir before running the tests
cd ${TEMP_BUILD_DIR}

for f in ${PARTICLE_LIBRARIES_DIR}/tensorflow_lite/examples/*/*.ino; do
    ${PARTICLE_CLI_TOOL} compile xenon $f
done

# rm -rf ${PARTICLE_LIBRARIES_DIR}
