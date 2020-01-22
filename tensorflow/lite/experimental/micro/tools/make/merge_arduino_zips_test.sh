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

INPUT1_DIR=${TEST_TMPDIR}/input1
mkdir -p ${INPUT1_DIR}
touch ${INPUT1_DIR}/a.txt
touch ${INPUT1_DIR}/b.txt
mkdir ${INPUT1_DIR}/sub1/
touch ${INPUT1_DIR}/sub1/c.txt
mkdir ${INPUT1_DIR}/sub2/
touch ${INPUT1_DIR}/sub2/d.txt
INPUT1_ZIP=${TEST_TMPDIR}/input1.zip
pushd ${INPUT1_DIR}
zip -q -r ${INPUT1_ZIP} *
popd

INPUT2_DIR=${TEST_TMPDIR}/input2
mkdir -p ${INPUT2_DIR}
touch ${INPUT2_DIR}/a.txt
touch ${INPUT2_DIR}/e.txt
mkdir ${INPUT2_DIR}/sub1/
touch ${INPUT2_DIR}/sub1/f.txt
mkdir ${INPUT2_DIR}/sub3/
touch ${INPUT2_DIR}/sub3/g.txt
INPUT2_ZIP=${TEST_TMPDIR}/input2.zip
pushd ${INPUT2_DIR}
zip -q -r ${INPUT2_ZIP} *
popd

OUTPUT_DIR=${TEST_TMPDIR}/output/
OUTPUT_ZIP=${OUTPUT_DIR}/output.zip

${TEST_SRCDIR}/tensorflow/lite/experimental/micro/tools/make/merge_arduino_zips \
  ${OUTPUT_ZIP} ${INPUT1_ZIP} ${INPUT2_ZIP}

if [[ ! -f ${OUTPUT_ZIP} ]]; then
  echo "${OUTPUT_ZIP} wasn't created."
fi

pushd ${OUTPUT_DIR}
unzip -q ${OUTPUT_ZIP}
popd

for EXPECTED_FILE in a.txt b.txt sub1/c.txt sub2/d.txt e.txt sub1/f.txt sub3/g.txt
do
  if [[ ! -f ${OUTPUT_DIR}/${EXPECTED_FILE} ]]; then
    echo "${OUTPUT_DIR}/${EXPECTED_FILE} wasn't created."
    exit 1
  fi
done

echo
echo "SUCCESS: merge_arduino_zips test PASSED"
