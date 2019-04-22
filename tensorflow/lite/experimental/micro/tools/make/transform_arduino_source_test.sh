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

INPUT_TEMPLATE=${TEST_SRCDIR}/tensorflow/lite/experimental/micro/tools/make/templates/keil_project.uvprojx.tpl

INPUT_FILE=${TEST_TMPDIR}/input_example.cc
cat << EOF > ${INPUT_FILE}
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

OUTPUT_FILE=${TEST_TMPDIR}/output_example.cc
THIRD_PARTY_HEADERS="subdir/foo.h subdir_2/include/bar/fish.h subdir_3/something.h"

${TEST_SRCDIR}/tensorflow/lite/experimental/micro/tools/make/transform_arduino_source \
  --third_party_headers="${THIRD_PARTY_HEADERS}" \
  < ${INPUT_FILE} \
  > ${OUTPUT_FILE}

if ! grep -q '#include <stdio.h>' ${OUTPUT_FILE}; then
  echo "ERROR: No stdio.h include found in output '${OUTPUT_FILE}'"
  exit 1
fi

if ! grep -q '#include "subdir/foo.h"' ${OUTPUT_FILE}; then
  echo "ERROR: No subdir/foo.h include found in output '${OUTPUT_FILE}'"
  exit 1
fi

if ! grep -q '#include "subdir_2/include/bar/fish.h"' ${OUTPUT_FILE}; then
  echo "ERROR: No subdir_2/include/bar/fish.h include found in output '${OUTPUT_FILE}'"
  exit 1
fi

if ! grep -q '#include "baz.h"' ${OUTPUT_FILE}; then
  echo "ERROR: No baz.h include found in output '${OUTPUT_FILE}'"
  exit 1
fi

if ! grep -q '#include "subdir_3/something.h"' ${OUTPUT_FILE}; then
  echo "ERROR: No subdir_3/something.h include found in output '${OUTPUT_FILE}'"
  exit 1
fi

echo
echo "SUCCESS: transform_arduino_source test PASSED"
