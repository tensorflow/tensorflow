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

INPUT_TEMPLATE=${TEST_SRCDIR}/tensorflow/lite/micro/tools/make/templates/keil_project.uvprojx.tpl
OUTPUT_FILE=${TEST_TMPDIR}/keil_project.uvprojx
EXECUTABLE=test_executable

${TEST_SRCDIR}/tensorflow/lite/micro/tools/make/generate_keil_project \
  --input_template=${INPUT_TEMPLATE} \
  --output_file=${OUTPUT_FILE} \
  --executable=${EXECUTABLE} \
  --hdrs="foo.h bar.h" \
  --srcs="foo.c bar.cc some/bad<xml.cc" \
  --include_paths=". include"

if ! grep -q "${EXECUTABLE}" ${OUTPUT_FILE}; then
  echo "ERROR: No executable name '${EXECUTABLE}' found in project file '${OUTPUT_FILE}'."
  exit 1
fi

if ! grep -q "foo\.h" ${OUTPUT_FILE}; then
  echo "ERROR: No header 'foo.h' found in project file '${OUTPUT_FILE}'."
  exit 1
fi

if ! grep -q "bar\.h" ${OUTPUT_FILE}; then
  echo "ERROR: No header 'bar.h' found in project file '${OUTPUT_FILE}'."
  exit 1
fi

if ! grep -q "foo\.c" ${OUTPUT_FILE}; then
  echo "ERROR: No source 'foo.c' found in project file '${OUTPUT_FILE}'."
  exit 1
fi

if ! grep -q "bar\.cc" ${OUTPUT_FILE}; then
  echo "ERROR: No source 'bar.cc' found in project file '${OUTPUT_FILE}'."
  exit 1
fi

if ! grep -q "some/badxml\.cc" ${OUTPUT_FILE}; then
  echo "ERROR: No source 'some/badxml.cc' found in project file '${OUTPUT_FILE}'."
  exit 1
fi

if ! grep -q "\.;include" ${OUTPUT_FILE}; then
  echo "ERROR: No include paths '.;include' found in project file '${OUTPUT_FILE}'."
  exit 1
fi

echo
echo "SUCCESS: generate_keil_project test PASSED"
