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
#
# To select only a subset of the Python tests to run, set the environment
# variable PY_TEST_WHITELIST, e.g.,
#   PY_TEST_WHITELIST="tensorflow/python/kernel_tests/shape_ops_test.py"
#
# Seprate the tests with a space. Leave this environment variable empty to
# disable the whitelist.

# You can also ignore a set of the tests by using the variable PY_TEST_BLACKLIST
# These are the test that depend on Python modules in the source that are not
# exported publicly:
PY_TEST_BLACKLIST="tensorflow/python/tools/freeze_graph_test.py "\
"tensorflow/python/util/protobuf/compare_test.py "\
"tensorflow/python/framework/ops_test.py "\
"tensorflow/python/framework/device_test.py "

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
COUNTER=0
SKIP_COUNTER=0
for TEST_FILE_PATH in ${ALL_PY_TESTS}; do
  ((COUNTER++))

  # Copy to a separate directory to guard against the possibility of picking up 
  # modules in the source directory 
  cp ${TEST_FILE_PATH} ${PY_TEST_DIR}/ &&

  # If PY_TEST_WHITELIST is not empty, only the white-listed tests will be run
  if [[ ! -z ${PY_TEST_WHITELIST} ]] && \
     [[ ! ${PY_TEST_WHITELIST} == *"${TEST_FILE_PATH}"* ]]; then
    ((SKIP_COUNTER++))
    echo "Skipping non-whitelisted test: ${TEST_FILE_PATH}"
    continue
  fi

  # If the test is in the black list, skip it
  if [[ ${PY_TEST_BLACKLIST} == *"${TEST_FILE_PATH}"* ]]; then
    ((SKIP_COUNTER++))
    echo "Skipping blacklisted test: ${TEST_FILE_PATH}"
    continue
  fi

  echo "==============================================================="
  echo "Running Python test on install #${COUNTER} of ${PY_TEST_COUNT}:"
  echo "  ${TEST_FILE_PATH}"
  echo "==============================================================="
  echo ""

  TEST_FILE_BASENAME=`basename "${TEST_FILE_PATH}"`

  TEST_OUTPUT=${PY_TEST_DIR}/${TEST_FILE_BASENAME}.out

  python ${PY_TEST_DIR}/${TEST_FILE_BASENAME} 2>&1 | \
    tee ${TEST_OUTPUT}  &&

  # Check for OK or failure status of the test output and exit with code 1
  # if failure is seen
  OK_LINE=`grep "^OK" ${TEST_OUTPUT}`
  FAIL_LINE=`grep "^> FAILED" ${TEST_OUTPUT}`
  if [[ ! -z ${OK_LINE} ]] && [[ -z ${FAIL_LINE} ]]; then
    echo ""
    echo "Python test on install succeeded: ${TEST_FILE_PATH}"
    echo ""
  else
    echo ""
    echo "Python test on install FAILED: ${TEST_FILE_PATH}"
    exit 1
  fi

  # Clean up files for this test
  rm -f ${PY_TEST_DIR}/${TEST_FILE_BASENAME} &&
  rm -f ${TEST_OUTPUT}

done

echo ""
echo "${PY_TEST_COUNT} Python tests on install PASSED" \
     "(Among those, ${SKIP_COUNTER} were SKIPPED)"
