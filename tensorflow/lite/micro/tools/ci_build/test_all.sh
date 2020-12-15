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

make -f tensorflow/lite/micro/tools/make/Makefile clean_downloads DISABLE_DOWNLOADS=true


make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn clean DISABLE_DOWNLOADS=true
if [ -d tensorflow/lite/micro/tools/make/downloads ]; then
  echo "ERROR: Downloads directory should not exist, but it does."
  exit 1
fi

# explicitly call third_party_downloads since we need pigweed for the license
# and clang-format checks.
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

# Check for license with the necessary exclusions.
tensorflow/lite/micro/tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/pigweed_presubmit.py \
  tensorflow/lite/micro \
  -p copyright_notice \
  -e micro/tools/make/targets/ecm3531 \
  -e BUILD\
  -e leon_commands \
  -e Makefile \
  -e "\.bzl" \
  -e "\.cmd" \
  -e "\.conf" \
  -e "\.defaults" \
  -e "\.h5" \
  -e "\.ipynb" \
  -e "\.inc" \
  -e "\.lcf" \
  -e "\.ld" \
  -e "\.lds" \
  -e "\.patch" \
  -e "\.projbuild" \
  -e "\.properties" \
  -e "\.resc" \
  -e "\.robot" \
  -e "\.txt" \
  -e "\.tpl" \
  --output-directory /tmp

# Check that the TFLM-only code is clang-formatted We are currently ignoring
# Python files (with yapf as the formatter) because that needs additional setup.
tensorflow/lite/micro/tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/format_code.py \
  tensorflow/lite/micro \
  -e "\.inc" \
  -e "\.py"


# We are moving away from having the downloads and installations be part of the
# Makefile. As a result, we need to manually add the downloads in this script.
# Once we move more than the renode downloads out of the Makefile, we should
# have common way to perform the downloads for a given target, tags ...
echo "Starting renode download at `date`"
tensorflow/lite/micro/testing/download_renode.sh tensorflow/lite/micro/tools/make/downloads/renode
pip3 install -r tensorflow/lite/micro/tools/make/downloads/renode/tests/requirements.txt

# Add all the test scripts for the various supported platforms here. This
# enables running all the tests together has part of the continuous integration
# pipeline and reduces duplication associated with setting up the docker
# environment.

echo "Starting to run micro tests at `date`"

echo "Running x86 tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_x86.sh PRESUBMIT

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
