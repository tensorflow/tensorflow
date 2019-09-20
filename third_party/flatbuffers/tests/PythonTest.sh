#!/bin/bash -eu
#
# Copyright 2014 Google Inc. All rights reserved.
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

pushd "$(dirname $0)" >/dev/null
test_dir="$(pwd)"
gen_code_path=${test_dir}
runtime_library_dir=${test_dir}/../python

# Emit Python code for the example schema in the test dir:
${test_dir}/../flatc -p -o ${gen_code_path} -I include_test monster_test.fbs

# Syntax: run_tests <interpreter> <benchmark vtable dedupes>
#                   <benchmark read count> <benchmark build count>
interpreters_tested=()
function run_tests() {
  if $(which ${1} >/dev/null); then
    echo "Testing with interpreter: ${1}"
    PYTHONDONTWRITEBYTECODE=1 \
    JYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=${runtime_library_dir}:${gen_code_path} \
    JYTHONPATH=${runtime_library_dir}:${gen_code_path} \
    COMPARE_GENERATED_TO_GO=0 \
    COMPARE_GENERATED_TO_JAVA=0 \
    $1 py_test.py $2 $3 $4
    interpreters_tested+=(${1})
    echo
  fi
}

# Run test suite with these interpreters. The arguments are benchmark counts.
run_tests python2.6 100 100 100
run_tests python2.7 100 100 100
run_tests python3 100 100 100
run_tests pypy 100 100 100

# NOTE: We'd like to support python2.5 in the future.

# NOTE: Jython 2.7.0 fails due to a bug in the stdlib `struct` library:
#       http://bugs.jython.org/issue2188

if [ ${#interpreters_tested[@]} -eq 0 ]; then
  echo "No Python interpeters found on this system, could not run tests."
  exit 1
fi

# Run test suite with default python intereter.
# (If the Python program `coverage` is available, it will be run, too.
#  Install `coverage` with `pip install coverage`.)
if $(which coverage >/dev/null); then
  echo 'Found coverage utility, running coverage with default Python:'

  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=${runtime_library_dir}:${gen_code_path} \
  coverage run --source=flatbuffers,MyGame py_test.py 0 0 0 > /dev/null

  echo
  cov_result=`coverage report --omit="*flatbuffers/vendor*,*py_test*" \
              | tail -n 1 | awk ' { print $4 } '`
  echo "Code coverage: ${cov_result}"
else
  echo -n "Did not find coverage utility for default Python, skipping. "
  echo "Install with 'pip install coverage'."
fi

echo
echo "OK: all tests passed for ${#interpreters_tested[@]} interpreters: ${interpreters_tested[@]}."
