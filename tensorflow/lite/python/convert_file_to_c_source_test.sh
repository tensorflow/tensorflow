#!/bin/bash
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
# Bash unit tests for the TensorFlow Lite Micro project generator.

set -e

INPUT_FILE=${TEST_TMPDIR}/input.tflite
printf "\x00\x01\x02\x03" > ${INPUT_FILE}

OUTPUT_SOURCE_FILE=${TEST_TMPDIR}/output_source.cc
OUTPUT_HEADER_FILE=${TEST_TMPDIR}/output_header.h

# Needed for copybara compatibility.
SCRIPT_BASE_DIR=/org_"tensor"flow
${TEST_SRCDIR}${SCRIPT_BASE_DIR}/tensorflow/lite/python/convert_file_to_c_source \
  --input_tflite_file="${INPUT_FILE}" \
  --output_source_file="${OUTPUT_SOURCE_FILE}" \
  --output_header_file="${OUTPUT_HEADER_FILE}" \
  --array_variable_name="g_some_array" \
  --line_width=80 \
  --include_guard="SOME_GUARD_H_" \
  --include_path="some/guard.h" \
  --use_tensorflow_license=True

if ! grep -q 'const unsigned char g_some_array' ${OUTPUT_SOURCE_FILE}; then
  echo "ERROR: No array found in output '${OUTPUT_SOURCE_FILE}'"
  exit 1
fi

if ! grep -q '0x00, 0x01, 0x02, 0x03' ${OUTPUT_SOURCE_FILE}; then
  echo "ERROR: No array values found in output '${OUTPUT_SOURCE_FILE}'"
  exit 1
fi

if ! grep -q 'const int g_some_array_len = 4;' ${OUTPUT_SOURCE_FILE}; then
  echo "ERROR: No array length found in output '${OUTPUT_SOURCE_FILE}'"
  exit 1
fi

if ! grep -q 'The TensorFlow Authors. All Rights Reserved' ${OUTPUT_SOURCE_FILE}; then
  echo "ERROR: No license found in output '${OUTPUT_SOURCE_FILE}'"
  exit 1
fi

if ! grep -q '\#include "some/guard\.h"' ${OUTPUT_SOURCE_FILE}; then
  echo "ERROR: No include found in output '${OUTPUT_SOURCE_FILE}'"
  exit 1
fi


if ! grep -q '#ifndef SOME_GUARD_H_' ${OUTPUT_HEADER_FILE}; then
  echo "ERROR: No include guard found in output '${OUTPUT_HEADER_FILE}'"
  exit 1
fi

if ! grep -q 'extern const unsigned char g_some_array' ${OUTPUT_HEADER_FILE}; then
  echo "ERROR: No array found in output '${OUTPUT_HEADER_FILE}'"
  exit 1
fi

if ! grep -q 'extern const int g_some_array_len;' ${OUTPUT_HEADER_FILE}; then
  echo "ERROR: No array length found in output '${OUTPUT_HEADER_FILE}'"
  exit 1
fi

if ! grep -q 'The TensorFlow Authors. All Rights Reserved' ${OUTPUT_HEADER_FILE}; then
  echo "ERROR: No license found in output '${OUTPUT_HEADER_FILE}'"
  exit 1
fi


echo
echo "SUCCESS: convert_file_to_c_source test PASSED"
