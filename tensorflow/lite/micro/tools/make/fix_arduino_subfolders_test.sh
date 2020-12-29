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

LIBRARY_DIR=${TEST_TMPDIR}/library
mkdir -p ${LIBRARY_DIR}

EXAMPLES_SUBDIR_CPP=${LIBRARY_DIR}/examples/something/foo/fish.cpp
mkdir -p `dirname ${EXAMPLES_SUBDIR_CPP}`
touch ${EXAMPLES_SUBDIR_CPP}

EXAMPLES_SUBDIR_HEADER=${LIBRARY_DIR}/examples/something/foo/fish.h
mkdir -p `dirname ${EXAMPLES_SUBDIR_HEADER}`
touch ${EXAMPLES_SUBDIR_HEADER}

TENSORFLOW_SRC_DIR=${LIBRARY_DIR}/src/
PERSON_DATA_FILE=${TENSORFLOW_SRC_DIR}tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_detect_model_data.cpp
mkdir -p `dirname ${PERSON_DATA_FILE}`
echo '#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"' > ${PERSON_DATA_FILE}
mkdir -p ${LIBRARY_DIR}/examples/person_detection

EXAMPLE_INO_FILE=${LIBRARY_DIR}/examples/something/main.ino
mkdir -p `dirname ${EXAMPLE_INO_FILE}`
touch ${EXAMPLE_INO_FILE}

${TEST_SRCDIR}/tensorflow/lite/micro/tools/make/fix_arduino_subfolders \
  ${LIBRARY_DIR}

EXPECTED_EXAMPLES_SUBDIR_CPP=${LIBRARY_DIR}/examples/something/foo_fish.cpp
if [[ ! -f ${EXPECTED_EXAMPLES_SUBDIR_CPP} ]]; then
  echo "${EXPECTED_EXAMPLES_SUBDIR_CPP} wasn't created."
  exit 1
fi

EXPECTED_EXAMPLES_SUBDIR_HEADER=${LIBRARY_DIR}/examples/something/foo_fish.h
if [[ ! -f ${EXPECTED_EXAMPLES_SUBDIR_HEADER} ]]; then
  echo "${EXPECTED_EXAMPLES_SUBDIR_HEADER} wasn't created."
  exit 1
fi

EXPECTED_PERSON_DATA_FILE=${LIBRARY_DIR}/examples/person_detection/person_detect_model_data.cpp
if [[ ! -f ${EXPECTED_PERSON_DATA_FILE} ]]; then
  echo "${EXPECTED_PERSON_DATA_FILE} wasn't created."
  exit 1
fi

if ! grep -q '#include "person_detect_model_data.h"' ${EXPECTED_PERSON_DATA_FILE}; then
  echo "ERROR: No person_detect_model_data.h include found in output '${EXPECTED_PERSON_DATA_FILE}'"
  exit 1
fi

EXPECTED_EXAMPLE_INO_FILE=${LIBRARY_DIR}/examples/something/something.ino
if [[ ! -f ${EXPECTED_EXAMPLE_INO_FILE} ]]; then
  echo "${EXPECTED_EXAMPLE_INO_FILE} wasn't created."
  exit 1
fi

echo
echo "SUCCESS: fix_arduino_subfolders test PASSED"
