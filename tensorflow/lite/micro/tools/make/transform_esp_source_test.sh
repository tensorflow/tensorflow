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

INPUT_EXAMPLE_FILE=${TEST_TMPDIR}/input_example.cc
cat << EOF > ${INPUT_EXAMPLE_FILE}
#include <stdio.h>
#include "baz.h"
#include "tensorflow/lite/micro/examples/something/foo/fish.h"

main() {
  fprintf(stderr, "Hello World!\n");
  return 0;
}
EOF

OUTPUT_EXAMPLE_FILE=${TEST_TMPDIR}/output_example.cc

${TEST_SRCDIR}/tensorflow/lite/micro/tools/make/transform_source \
  --platform=esp \
  --is_example_source \
  --source_path="tensorflow/lite/micro/examples/something/input_example.cc" \
  < ${INPUT_EXAMPLE_FILE} \
  > ${OUTPUT_EXAMPLE_FILE}

if ! grep -q '#include <stdio.h>' ${OUTPUT_EXAMPLE_FILE}; then
  echo "ERROR: No stdio.h include found in output '${OUTPUT_EXAMPLE_FILE}'"
  exit 1
fi

if ! grep -q '#include "baz.h"' ${OUTPUT_EXAMPLE_FILE}; then
  echo "ERROR: No baz.h include found in output '${OUTPUT_EXAMPLE_FILE}'"
  exit 1
fi

if ! grep -q '#include "foo/fish.h"' ${OUTPUT_EXAMPLE_FILE}; then
  echo "ERROR: No foo/fish.h include found in output '${OUTPUT_EXAMPLE_FILE}'"
  exit 1
fi


#
# Example file in a sub directory.
#

mkdir -p "${TEST_TMPDIR}/subdir"
INPUT_EXAMPLE_SUBDIR_FILE=${TEST_TMPDIR}/subdir/input_example.cc
cat << EOF > ${INPUT_EXAMPLE_SUBDIR_FILE}
#include <stdio.h>
#include "baz.h"
#include "tensorflow/lite/micro/examples/something/subdir/input_example.h"
#include "tensorflow/lite/micro/examples/something/bleh.h"
#include "tensorflow/lite/micro/examples/something/foo/fish.h"
EOF

OUTPUT_EXAMPLE_SUBDIR_FILE=${TEST_TMPDIR}/output_example.cc

${TEST_SRCDIR}/tensorflow/lite/micro/tools/make/transform_source \
  --platform=esp \
  --is_example_source \
  --source_path="tensorflow/lite/micro/examples/something/subdir/input_example.cc" \
  < ${INPUT_EXAMPLE_SUBDIR_FILE} \
  > ${OUTPUT_EXAMPLE_SUBDIR_FILE}

if ! grep -q '#include <stdio.h>' ${OUTPUT_EXAMPLE_SUBDIR_FILE}; then
  echo "ERROR: No stdio.h include found in output '${OUTPUT_EXAMPLE_SUBDIR_FILE}'"
  exit 1
fi

if ! grep -q '#include "baz.h"' ${OUTPUT_EXAMPLE_SUBDIR_FILE}; then
  echo "ERROR: No baz.h include found in output '${OUTPUT_EXAMPLE_SUBDIR_FILE}'"
  exit 1
fi

if ! grep -q '#include "input_example.h"' ${OUTPUT_EXAMPLE_SUBDIR_FILE}; then
  echo "ERROR: No input_example.h include found in output '${OUTPUT_EXAMPLE_SUBDIR_FILE}'"
  cat ${OUTPUT_EXAMPLE_SUBDIR_FILE}
  exit 1
fi

if ! grep -q '#include "../bleh.h"' ${OUTPUT_EXAMPLE_SUBDIR_FILE}; then
  echo "ERROR: No ../bleh.h include found in output '${OUTPUT_EXAMPLE_SUBDIR_FILE}'"
  exit 1
fi

if ! grep -q '#include "../foo/fish.h"' ${OUTPUT_EXAMPLE_SUBDIR_FILE}; then
  echo "ERROR: No ../foo/fish.h include found in output '${OUTPUT_EXAMPLE_SUBDIR_FILE}'"
  exit 1
fi

echo
echo "SUCCESS: transform_esp_source test PASSED"
