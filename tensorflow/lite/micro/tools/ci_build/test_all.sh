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
cd "${ROOT_DIR}"
pwd

echo "Starting to run micro tests at `date`"

make -f tensorflow/lite/micro/tools/make/Makefile clean_downloads DISABLE_DOWNLOADS=true
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn clean DISABLE_DOWNLOADS=true
if [ -d tensorflow/lite/micro/tools/make/downloads ]; then
  echo "ERROR: Downloads directory should not exist, but it does."
  exit 1
fi

echo "Running code style checks at `date`"
tensorflow/lite/micro/tools/ci_build/test_code_style.sh PRESUBMIT

# Add all the test scripts for the various supported platforms here. This
# enables running all the tests together has part of the continuous integration
# pipeline and reduces duplication associated with setting up the docker
# environment.

if [[ ${1} == "GITHUB_PRESUBMIT" ]]; then
  # We enable bazel as part of the github CI only. This is because the same
  # checks are already part of the internal CI and there isn't a good reason to
  # duplicate them.
  #
  # Another reason is that the bazel checks involve some patching of TF
  # workspace and BUILD files and this is an experiment to see what the
  # trade-off should be between the maintenance overhead, increased CI time from
  # the unnecessary TF downloads.
  #
  # See https://github.com/tensorflow/tensorflow/issues/46465 and
  # http://b/177672856 for more context.
  echo "Running bazel tests at `date`"
  tensorflow/lite/micro/tools/ci_build/test_bazel.sh

  # Enabling FVP for github CI only. This is because it currently adds ~4mins to each
  # Kokoro run and is only relevant for external changes. Given all the other TFLM CI
  # coverage, it is unlikely that an internal change would break only the corstone build.
  echo "Running cortex_m_corstone_300 tests at `date`"
  tensorflow/lite/micro/tools/ci_build/test_cortex_m_corstone_300.sh

  # Only running project generation v2 prototype as part of the github CI while
  # it is under development. See
  # https://github.com/tensorflow/tensorflow/issues/47413 for more context.
  echo "Running project_generation test at `date`"
  tensorflow/lite/micro/tools/ci_build/test_project_generation.sh
fi

echo "Running x86 tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_x86.sh

echo "Running bluepill tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_bluepill.sh

# TODO(b/174189223): Skipping mbed tests due to:
# https://github.com/tensorflow/tensorflow/issues/45164
# echo "Running mbed tests at `date`"
# tensorflow/lite/micro/tools/ci_build/test_mbed.sh PRESUBMIT

echo "Running Sparkfun tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_sparkfun.sh

echo "Running stm32f4 tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_stm32f4.sh PRESUBMIT

echo "Running Arduino tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_arduino.sh

echo "Running cortex_m_generic tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_cortex_m_generic.sh

echo "Finished all micro tests at `date`"
