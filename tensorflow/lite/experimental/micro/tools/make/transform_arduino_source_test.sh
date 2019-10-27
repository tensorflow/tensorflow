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

INPUT_REGULAR_FILE=${TEST_TMPDIR}/input_regular.cc
cat << EOF > ${INPUT_REGULAR_FILE}
#include <stdio.h>
#include "foo.h"
#include "bar/fish.h"
#include "baz.h"
#ifndef __ANDROID__
  #include "something.h"
#endif  // __ANDROID__

int main(int argc, char** argv) {
  fprintf(stderr, "Hello World!\n");
  return 0;
}
EOF

OUTPUT_REGULAR_FILE=${TEST_TMPDIR}/output_regular.cc
THIRD_PARTY_HEADERS="subdir/foo.h subdir_2/include/bar/fish.h subdir_3/something.h"

${TEST_SRCDIR}/tensorflow/lite/experimental/micro/tools/make/transform_arduino_source \
  --third_party_headers="${THIRD_PARTY_HEADERS}" \
  < ${INPUT_REGULAR_FILE} \
  > ${OUTPUT_REGULAR_FILE}

if ! grep -q '#include <stdio.h>' ${OUTPUT_REGULAR_FILE}; then
  echo "ERROR: No stdio.h include found in output '${OUTPUT_REGULAR_FILE}'"
  exit 1
fi

if ! grep -q '#include "subdir/foo.h"' ${OUTPUT_REGULAR_FILE}; then
  echo "ERROR: No subdir/foo.h include found in output '${OUTPUT_REGULAR_FILE}'"
  exit 1
fi

if ! grep -q '#include "subdir_2/include/bar/fish.h"' ${OUTPUT_REGULAR_FILE}; then
  echo "ERROR: No subdir_2/include/bar/fish.h include found in output '${OUTPUT_REGULAR_FILE}'"
  exit 1
fi

if ! grep -q '#include "baz.h"' ${OUTPUT_REGULAR_FILE}; then
  echo "ERROR: No baz.h include found in output '${OUTPUT_REGULAR_FILE}'"
  exit 1
fi

if ! grep -q '#include "subdir_3/something.h"' ${OUTPUT_REGULAR_FILE}; then
  echo "ERROR: No subdir_3/something.h include found in output '${OUTPUT_REGULAR_FILE}'"
  exit 1
fi

if ! grep -q 'int tflite_micro_main(' ${OUTPUT_REGULAR_FILE}; then
  echo "ERROR: No int tflite_micro_main() definition found in output '${OUTPUT_REGULAR_FILE}'"
  exit 1
fi


INPUT_EXAMPLE_INO_FILE=${TEST_TMPDIR}/input_example_ino.cc
cat << EOF > ${INPUT_EXAMPLE_INO_FILE}
#include <stdio.h>
#include "foo.h"
#include "tensorflow/lite/experimental/micro/examples/something/foo/fish.h"
#include "baz.h"

void setup() {
}

void loop() {
}
EOF

OUTPUT_EXAMPLE_INO_FILE=${TEST_TMPDIR}/output_regular.cc

${TEST_SRCDIR}/tensorflow/lite/experimental/micro/tools/make/transform_arduino_source \
  --third_party_headers="${THIRD_PARTY_HEADERS}" \
  --is_example_ino \
  < ${INPUT_EXAMPLE_INO_FILE} \
  > ${OUTPUT_EXAMPLE_INO_FILE}

if ! grep -q '#include <TensorFlowLite.h>' ${OUTPUT_EXAMPLE_INO_FILE}; then
  echo "ERROR: No TensorFlowLite.h include found in output '${OUTPUT_EXAMPLE_INO_FILE}'"
  exit 1
fi

if ! grep -q '#include "foo_fish.h"' ${OUTPUT_EXAMPLE_INO_FILE}; then
  echo "ERROR: No foo/fish.h include found in output '${OUTPUT_EXAMPLE_INO_FILE}'"
  exit 1
fi

INPUT_EXAMPLE_SOURCE_FILE=${TEST_TMPDIR}/input_example_source.h
cat << EOF > ${INPUT_EXAMPLE_SOURCE_FILE}
#include <stdio.h>
#include "foo.h"
#include "foo/fish.h"
#include "baz.h"
#include "tensorflow/lite/experimental/micro/examples/something/cube/tri.h"

void setup() {
}

void loop() {
}

int main(int argc, char* argv[]) {
  setup();
  while (true) {
    loop();
  }
}
EOF

OUTPUT_EXAMPLE_SOURCE_FILE=${TEST_TMPDIR}/output_example_source.h

${TEST_SRCDIR}/tensorflow/lite/experimental/micro/tools/make/transform_arduino_source \
  --third_party_headers="${THIRD_PARTY_HEADERS}" \
  --is_example_source \
  --source_path="foo/input_example_source.h" \
  < ${INPUT_EXAMPLE_SOURCE_FILE} \
  > ${OUTPUT_EXAMPLE_SOURCE_FILE}

if ! grep -q '#include "foo/fish.h"' ${OUTPUT_EXAMPLE_SOURCE_FILE}; then
  echo "ERROR: No foo/fish.h include found in output '${OUTPUT_EXAMPLE_SOURCE_FILE}'"
  exit 1
fi

if ! grep -q '#include "cube_tri.h"' ${OUTPUT_EXAMPLE_SOURCE_FILE}; then
  echo "ERROR: No cube_tri.h include found in output '${OUTPUT_EXAMPLE_SOURCE_FILE}'"
  exit 1
fi


echo
echo "SUCCESS: transform_arduino_source test PASSED"
