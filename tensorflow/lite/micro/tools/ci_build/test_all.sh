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
# Creates the project file distributions for the TensorFlow Lite Micro test and
# example targets aimed at embedded platforms.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd ${ROOT_DIR}
pwd

make -f tensorflow/lite/micro/tools/make/Makefile \
  clean clean_downloads

# Add all the test scripts for the various supported platforms here. This
# emables running all the tests together has part of the continuous integration
# pipeline and reduces duplication associated with setting up the docker
# environment.

echo "Starting to run micro tests at `date`"

echo "Running Arduino tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_arduino.sh

echo "Running bluepill tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_bluepill.sh

echo "Running Sparkfun tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_sparkfun.sh

echo "Running x86 tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_x86.sh

echo "Finished all micro tests at `date`"
