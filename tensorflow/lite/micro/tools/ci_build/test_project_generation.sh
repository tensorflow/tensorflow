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

# This script can be used to initiate a bazel build with a reduced set of
# downloads, but still sufficient to test all the TFLM targets.
#
# This is primarily intended for use from a Docker image as part of the TFLM
# github continuous integration system. There are still a number of downloads
# (e.g. java) that are not necessary and it may be possible to further reduce
# the set of external libraries and downloads.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TEST_OUTPUT_DIR="/tmp/tflm_project_gen"
rm -rf ${TEST_OUTPUT_DIR}

TEST_OUTPUT_DIR_CMSIS="/tmp/tflm_project_gen_cmsis"
rm -rf ${TEST_OUTPUT_DIR_CMSIS}

readable_run \
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  ${TEST_OUTPUT_DIR} \
  -e hello_world

readable_run cp tensorflow/lite/micro/tools/project_generation/Makefile ${TEST_OUTPUT_DIR}

pushd ${TEST_OUTPUT_DIR} > /dev/null
readable_run make -j8 examples
popd > /dev/null

readable_run python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=cortex-m4" \
  ${TEST_OUTPUT_DIR_CMSIS}
