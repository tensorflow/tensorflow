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
# Common Bash functions used by build scripts


die() {
  # Print a message and exit with code 1.
  #
  # Usage: die <error_message>
  #   e.g., die "Something bad happened."

  echo $@
  exit 1
}

realpath() {
  # Get the real path of a file
  # Usage: realpath <file_path>

  if [[ $# != "1" ]]; then
    die "realpath: incorrect usage"
  fi

  [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

to_lower () {
  # Convert string to lower case.
  # Usage: to_lower <string>

  echo "$1" | tr '[:upper:]' '[:lower:]'
}

calc_elapsed_time() {
  # Calculate elapsed time. Takes nanosecond format input of the kind output
  # by date +'%s%N'
  #
  # Usage: calc_elapsed_time <START_TIME> <END_TIME>

  if [[ $# != "2" ]]; then
    die "calc_elapsed_time: incorrect usage"
  fi

  START_TIME=$1
  END_TIME=$2

  if [[ ${START_TIME} == *"N" ]]; then
    # Nanosecond precision not available
    START_TIME=$(echo ${START_TIME} | sed -e 's/N//g')
    END_TIME=$(echo ${END_TIME} | sed -e 's/N//g')
    ELAPSED="$(expr ${END_TIME} - ${START_TIME}) s"
  else
    ELAPSED="$(expr $(expr ${END_TIME} - ${START_TIME}) / 1000000) ms"
  fi

  echo ${ELAPSED}
}

run_in_directory() {
  # Copy the test script to a destination directory and run the test there.
  # Write test log to a log file.
  #
  # Usage: run_in_directory <DEST_DIR> <LOG_FILE> <TEST_SCRIPT>
  #                         [ARGS_FOR_TEST_SCRIPT]

  if [[ $# -lt "3" ]]; then
    die "run_in_directory: incorrect usage"
  fi

  DEST_DIR="$1"
  LOG_FILE="$2"
  TEST_SCRIPT="$3"
  shift 3
  SCRIPT_ARGS=("$@")

  # Get the absolute path of the log file
  LOG_FILE_ABS=$(realpath "${LOG_FILE}")

  cp "${TEST_SCRIPT}" "${DEST_DIR}"/
  SCRIPT_BASENAME=$(basename "${TEST_SCRIPT}")

  if [[ ! -f "${DEST_DIR}/${SCRIPT_BASENAME}" ]]; then
    echo "FAILED to copy script ${TEST_SCRIPT} to temporary directory "\
"${DEST_DIR}"
    return 1
  fi

  pushd "${DEST_DIR}" > /dev/null

  "${TIMEOUT_BIN}" --preserve-status ${TIMEOUT} \
    "${PYTHON_BIN_PATH}" "${SCRIPT_BASENAME}" ${SCRIPT_ARGS[@]} 2>&1 \
    > "${LOG_FILE_ABS}"

  rm -f "${SCRIPT_BASENAME}"
  popd > /dev/null

  if [[ $? != 0 ]]; then
    echo "Test \"${SCRIPT_BASENAME}\" FAILED"
    return 1
  fi

  return 0
}


test_runner() {
  # Run a suite of tests, print failure logs (if any), wall-time each test,
  # and show the summary at the end.
  #
  # Usage: test_runner <TEST_DESC> <ALL_TESTS> <TEST_BLACKLIST> <LOGS_DIR>
  # e.g.,  test_runner "Tutorial test-on-install" \
  #                    "test1 test2 test3" "test2 test3" "/tmp/log_dir"

  if [[ $# != "4" ]]; then
    die "test_runner: incorrect usage"
  fi

  TEST_DESC=$1
  ALL_TESTS_STR=$2
  TEST_BLACKLIST_SR=$3
  LOGS_DIR=$4

  NUM_TESTS=$(echo "${ALL_TESTS_STR}" | wc -w)
  ALL_TESTS=(${ALL_TESTS_STR})

  COUNTER=0
  PASSED_COUNTER=0
  FAILED_COUNTER=0
  FAILED_TESTS=""
  FAILED_TEST_LOGS=""
  SKIPPED_COUNTER=0
  for CURR_TEST in ${ALL_TESTS[@]}; do
    ((COUNTER++))
    STAT_STR="(${COUNTER} / ${NUM_TESTS})"

    if [[ "${TEST_BLACKLIST_STR}" == *"${CURR_TEST}"* ]]; then
      ((SKIPPED_COUNTER++))
      echo "${STAT_STR} Blacklisted ${TEST_DESC} SKIPPED: ${CURR_TEST}"
      continue
    fi

    START_TIME=$(date +'%s%N')

    LOG_FILE="${LOGS_DIR}/${CURR_TEST}.log"
    rm -rf ${LOG_FILE} ||
    die "Unable to remove existing log file: ${LOG_FILE}"

    "test_${CURR_TEST}" "${LOG_FILE}"
    TEST_RESULT=$?

    END_TIME=$(date +'%s%N')
    ELAPSED_TIME=$(calc_elapsed_time "${START_TIME}" "${END_TIME}")

    if [[ ${TEST_RESULT} == 0 ]]; then
      ((PASSED_COUNTER++))
      echo "${STAT_STR} ${TEST_DESC} PASSED: ${CURR_TEST} "\
  "(Elapsed time: ${ELAPSED_TIME})"
    else
      ((FAILED_COUNTER++))
      FAILED_TESTS="${FAILED_TESTS} ${CURR_TEST}"
      FAILED_TEST_LOGS="${FAILED_TEST_LOGS} ${LOG_FILE}"

      echo "${STAT_STR} ${TEST_DESC} FAILED: ${CURR_TEST} "\
  "(Elapsed time: ${ELAPSED_TIME})"

      echo "============== BEGINS failure log content =============="
      cat ${LOG_FILE}
      echo "============== ENDS failure log content =============="
      echo ""
    fi
  done

  echo "${NUM_TUT_TESTS} ${TEST_DESC} test(s): "\
  "${PASSED_COUNTER} passed; ${FAILED_COUNTER} failed; ${SKIPPED_COUNTER} skipped"

  if [[ ${FAILED_COUNTER} -eq 0  ]]; then
    echo ""
    echo "${TEST_DESC} SUCCEEDED"

    exit 0
  else
    echo "FAILED test(s):"
    FAILED_TEST_LOGS=($FAILED_TEST_LOGS)
    FAIL_COUNTER=0
    for TEST_NAME in ${FAILED_TESTS}; do
      echo "  ${TEST_DESC} (Log @: ${FAILED_TEST_LOGS[${FAIL_COUNTER}]})"
      ((FAIL_COUNTER++))
    done

    echo ""
    die "${TEST_DESC} FAILED"
  fi
}

configure_android_workspace() {
  # Modify the WORKSPACE file.
  # Note: This is workaround. This should be done by bazel.
  if grep -q '^android_sdk_repository' WORKSPACE && grep -q '^android_ndk_repository' WORKSPACE; then
    echo "You probably have your WORKSPACE file setup for Android."
  else
    if [ -z "${ANDROID_API_LEVEL}" -o -z "${ANDROID_BUILD_TOOLS_VERSION}" ] || \
        [ -z "${ANDROID_SDK_HOME}" -o -z "${ANDROID_NDK_HOME}" ]; then
      echo "ERROR: Your WORKSPACE file does not seems to have proper android"
      echo "       configuration and not all the environment variables expected"
      echo "       inside ci_build android docker container are set."
      echo "       Please configure it manually. See: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/README.md"
    else
      cat << EOF >> WORKSPACE
android_sdk_repository(
    name = "androidsdk",
    api_level = ${ANDROID_API_LEVEL},
    build_tools_version = "${ANDROID_BUILD_TOOLS_VERSION}",
    path = "${ANDROID_SDK_HOME}",
)

android_ndk_repository(
    name="androidndk",
    path="${ANDROID_NDK_HOME}",
    api_level=21)
EOF
    fi
  fi
}
