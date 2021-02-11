#!/usr/bin/env bash
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
# Tests the microcontroller code for esp32 platform

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"
pwd

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET=esp

# setup esp-idf and toolchains
echo "Checking out esp-idf..."
readable_run git clone --recursive --single-branch --branch release/v4.2 https://github.com/espressif/esp-idf.git
export IDF_PATH="${ROOT_DIR}"/esp-idf
cd $IDF_PATH
readable_run ./install.sh
readable_run . ./export.sh
cd "${ROOT_DIR}"

# clean all
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

# generate examples
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} generate_hello_world_esp_project
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} generate_person_detection_esp_project
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} generate_micro_speech_esp_project

# build examples
cd "${ROOT_DIR}"/tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32/prj/hello_world/esp-idf
readable_run idf.py build

cd "${ROOT_DIR}"/tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32/prj/person_detection/esp-idf
readable_run git clone https://github.com/espressif/esp32-camera.git components/esp32-camera
cd components/esp32-camera/
readable_run git checkout eacd640b8d379883bff1251a1005ebf3cf1ed95c
cd ../../
readable_run idf.py build

cd "${ROOT_DIR}"/tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32/prj/micro_speech/esp-idf
readable_run idf.py build
