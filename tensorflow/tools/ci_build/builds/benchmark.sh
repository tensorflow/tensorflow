#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Runs benchmark tests.
# After the completion of each benchmark test, the script calls a hook binary,
# specified with the environment variable TF_BUILD_BENCHMARK_HOOK, to handle
# the test log file. This hook binary may perform operations such as entering
# the test results into a database.
#
# Usage: benchmark [-c opt]
# Option flags
#    -c opt:  Use optimized C++ build ("-c opt")
#
# This script obeys the following environmental variables:
#   TF_BUILD_BENCHMARK_HOOK:
#     Path to a binary / script that will handle the test log and other related
#     info after the completion of each benchmark test.

set -u

echo ""
echo "====== Benchmark tests start ======"

# Process input arguments
OPT_FLAG=""
while getopts c: flag; do
  case ${flag} in
    c)
      if [[ ! -z "{OPTARG}" ]]; then
        OPT_FLAG="${OPT_FLAG} -c ${OPTARG}"
      fi
      ;;
  esac
done

BENCHMARK_HOOK=${TF_BUILD_BENCHMARK_HOOK:-""}


BENCHMARK_TAG="benchmark-test"
BENCHMARK_TESTS=$(bazel query \
    'attr("tags", "'"${BENCHMARK_TAG}"'", //tensorflow/...)')

if [[ -z "${BENCHMARK_TESTS}" ]]; then
  echo "ERROR: Cannot find any benchmark tests with the tag "\
"\"${BENCHMARK_TAG}\""
  exit 1
fi

N_TESTS=$(echo ${BENCHMARK_TESTS} | wc -w)

echo "Discovered ${N_TESTS} benchmark test(s) with the tag \"${BENCHMARK_TAG}\":"
echo ${BENCHMARK_TESTS}
echo ""

PASS_COUNTER=0
FAIL_COUNTER=0
FAILED_TESTS=""
COUNTER=0

# Iterate through the benchmark tests
for BENCHMARK_TEST in ${BENCHMARK_TESTS}; do
  ((COUNTER++))

  echo ""
  echo "Running benchmark test (${COUNTER} / ${N_TESTS}): ${BENCHMARK_TEST}"

  bazel test ${OPT_FLAG} --cache_test_results=no "${BENCHMARK_TEST}"
  TEST_RESULT=$?

  # Hook for database
  # Verify that test log exists
  TEST_LOG=$(echo ${BENCHMARK_TEST} |  sed -e 's/:/\//g')
  TEST_LOG="bazel-testlogs/${TEST_LOG}/test.log"
  if [[ -f "${TEST_LOG}" ]]; then
    echo "Benchmark ${BENCHMARK_TEST} done: log @ ${TEST_LOG}"

    # Call database hook if exists
    if [[ ! -z "${BENCHMARK_HOOK}" ]]; then
      # Assume that the hook binary/script takes two arguments:
      #   Argument 1: Compilation flags such as "-c opt" as a whole
      #   Argument 2: Test log containing the serialized TestResults proto

      echo "Calling database hook: ${TF_BUILD_BENCHMARK_LOG_HOOK} "\
"${OPT_FLAG} ${TEST_LOG}"

      ${TF_BUILD_BENCHMARK_LOG_HOOK} "${OPT_FLAG}" "${TEST_LOG}"
    else
      echo "WARNING: No hook binary is specified to handle test log ${TEST_LOG}"
    fi
  else
    # Mark as failure if the test log file cannot be found
    TEST_RESULT=2

    echo "ERROR: Cannot find log file from benchmark ${BENCHMARK_TEST} @ "\
"${TEST_LOG}"
  fi

  echo ""
  if [[ ${TEST_RESULT} -eq 0 ]]; then
    ((PASS_COUNTER++))

    echo "Benchmark test PASSED: ${BENCHMARK_TEST}"
  else
    ((FAIL_COUNTER++))

    FAILED_TESTS="${FAILED_TESTS} ${BENCHMARK_TEST}"

    echo "Benchmark test FAILED: ${BENCHMARK_TEST}"

    if [[ -f "${TEST_LOG}" ]]; then
      echo "============== BEGINS failure log content =============="
      cat ${TEST_LOG} >&2
      echo "============== ENDS failure log content =============="
      echo ""
    fi
  fi

done

# Summarize test results
echo ""
echo "${N_TESTS} Benchmark test(s):" \
     "${PASS_COUNTER} passed;" \
     "${FAIL_COUNTER} failed"

if [[ ${FAIL_COUNTER} -eq 0  ]]; then
  echo ""
  echo "Benchmark tests SUCCEEDED"

  exit 0
else
  echo "FAILED benchmark test(s):"
  FAIL_COUNTER=0
  for TEST_NAME in ${FAILED_TESTS}; do
    echo "  ${TEST_NAME}"
    ((FAIL_COUNTER++))
  done

  echo ""
  echo "Benchmark tests FAILED"
  exit 1
fi
