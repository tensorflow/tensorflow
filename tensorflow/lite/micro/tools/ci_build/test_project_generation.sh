#!/usr/bin/env bash
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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TEST_OUTPUT_DIR=$(mktemp -d)

readable_run \
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  ${TEST_OUTPUT_DIR} \
  -e hello_world

# Confirm that print_src_files and print_dest_files output valid paths (and
# nothing else).
set +x
FILES="$(python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
           ${TEST_OUTPUT_DIR} \
           --print_src_files --print_dest_files --no_copy)"

readable_run ls ${FILES} > /dev/null

# Next, make sure that the output tree has all the files needed buld the
# examples.
readable_run cp tensorflow/lite/micro/tools/project_generation/Makefile ${TEST_OUTPUT_DIR}
pushd ${TEST_OUTPUT_DIR} > /dev/null
readable_run make -j8 examples
popd > /dev/null

rm -rf ${TEST_OUTPUT_DIR}

# Check that we can export a TFLM tree with additional makefile options.
TEST_OUTPUT_DIR_CMSIS=$(mktemp -d)
readable_run python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=cortex-m4" \
  ${TEST_OUTPUT_DIR_CMSIS}

rm -rf ${TEST_OUTPUT_DIR_CMSIS}

