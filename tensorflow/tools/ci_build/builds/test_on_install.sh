#!/usr/bin/env bash
# Copyright 2016 Google Inc. All Rights Reserved.
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

# Build the Python PIP installation package for TensorFlow
# and run the Python unit tests from the source code on the installation

cd /tensorflow &&

# Build the pip package
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package &&

# Install
rm -rf _python_build &&
mkdir _python_build &&
cd _python_build &&
ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* \
  . &&
ln -s ../tensorflow/tools/pip_package/* . &&
python setup.py develop &&

# Run tests
cd .. &&
ALL_PY_TESTS=`find tensorflow/python -name "*_test.py" | xargs` &&
PY_TEST_COUNT=`echo ${ALL_PY_TESTS} | wc -w`

PY_TEST_DIR=${HOME}/tf_py_tests

rm -rf ${PY_TEST_DIR} &&
mkdir ${PY_TEST_DIR} &&

# Iterate through all the Python unit test files using the installation
for TEST_FILE_PATH in ${ALL_PY_TESTS}; do
  # Copy to a separate directory to guard against the possibility of picking up 
  # modules in the source directory 
  cp ${TEST_FILE_PATH} ${PY_TEST_DIR}/ &&

  echo "Running Python test on install: ${TEST_FILE_PATH}"
  TEST_FILE_BASENAME=`basename "${TEST_FILE_PATH}"`

  TEST_OUTPUT=${PY_TEST_DIR}/${TEST_FILE_BASENAME}.out

  python ${PY_TEST_DIR}/${TEST_FILE_BASENAME} | \
    tee ${TEST_OUTPUT}  &&

  # Examine the OK status of the test output
  LASTLINE=`awk '/./{line=$0} END{print line}' ${TEST_OUTPUT}`
  test "${LASTLINE}" = OK &&

  # Clean up files for this test
  rm -f ${PY_TEST_DIR}/${TEST_FILE_BASENAME} &&
  rm -f ${PY_TEST_DIR}/${TEST_FILE_BASENAME}.out

done

echo ""
echo "All ${PY_TEST_COUNT} Python tests on install PASSED"
