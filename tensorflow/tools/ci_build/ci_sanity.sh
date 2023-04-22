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
# Usage: ci_sanity.sh [--pep8] [--incremental] [bazel flags]
#
# Options:
#           run sanity checks: python 2&3 pylint checks and bazel nobuild
#  --pep8   run pep8 test only
#  --pylint run pylint check only
#  --incremental  Performs checks incrementally, by using the files changed in
#                 the latest commit

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
source "${SCRIPT_DIR}/builds/builds_common.sh"

ROOT_DIR=$( cd "$SCRIPT_DIR/../../.." && pwd -P )

# Helper functions
die() {
  echo $@
  exit 1
}

num_cpus() {
  # Get the number of CPUs
  if [[ -f /proc/cpuinfo ]]; then
    N_CPUS=$(grep -c ^processor /proc/cpuinfo)
  else
    # Fallback method
    N_CPUS=`getconf _NPROCESSORS_ONLN`
  fi
  if [[ -z ${N_CPUS} ]]; then
    die "ERROR: Unable to determine the number of CPUs"
  fi

  echo ${N_CPUS}
}

# Helper functions for examining changed files in the last non-merge git
# commit.

# Get the hash of the last non-merge git commit on the current branch.
# Usage: get_last_non_merge_git_commit
get_last_non_merge_git_commit() {
  git rev-list --no-merges -n 1 HEAD
}

# List files changed (i.e., added, removed or revised) in the last non-merge
# git commit.
# Usage: get_changed_files_in_last_non_merge_git_commit
get_changed_files_in_last_non_merge_git_commit() {
  git diff-tree --no-commit-id --name-only -r $(get_last_non_merge_git_commit)
}

# List Python files changed in the last non-merge git commit that still exist,
# i.e., not removed.
# Usage: get_py_files_to_check [--incremental]
get_py_files_to_check() {
  if [[ "$1" == "--incremental" ]]; then
    CHANGED_PY_FILES=$(get_changed_files_in_last_non_merge_git_commit | \
                       grep '.*\.py$')

    # Do not include files removed in the last non-merge commit.
    PY_FILES=""
    for PY_FILE in ${CHANGED_PY_FILES}; do
      if [[ -f "${PY_FILE}" ]]; then
        PY_FILES="${PY_FILES} ${PY_FILE}"
      fi
    done

    echo "${PY_FILES}"
  else
    find tensorflow -name '*.py'
  fi
}

# Subfunctions for substeps
# Run pylint
do_pylint() {
  # Usage: do_pylint [--incremental]
  #
  # Options:
  #   --incremental  Performs check only if there are python files that changed
  #                  since last non-merge git commit. We always check all Python
  #                  files if one changed to capture the case when a function
  #                  signature changes affects unchanged files.

  # Validate arguments, see if we can do no work
  if [[ $# == 1 ]] && [[ "$1" == "--incremental" ]]; then
    PYTHON_SRC_FILES=$(get_py_files_to_check --incremental)

    if [[ -z "${PYTHON_SRC_FILES}" ]]; then
      echo "do_pylint will NOT run due to --incremental flag and due to the "\
"absence of Python code changes in the last commit."
      return 0
    fi
  elif [[ $# != 0 ]]; then
    echo "Invalid syntax for invoking do_pylint"
    echo "Usage: do_pylint [--incremental]"
    return 1
  else
    # Get all Python files, regardless of mode.
    PYTHON_SRC_FILES=$(get_py_files_to_check)
  fi

  # Something happened. TF no longer has Python code if this branch is taken
  if [[ -z ${PYTHON_SRC_FILES} ]]; then
    echo "do_pylint found no Python files to check. Returning."
    return 0
  fi

  # Now that we know we have to do work, check if `pylint` is installed
  PYLINT_BIN="python3.8 -m pylint"

  echo ""
  echo "print python version and pip freeze for debugging."
  echo ""
  python3.8
  python3.8 -m pip freeze

  echo ""
  echo "check whether pylint is available or not."
  echo ""

  ${PYLINT_BIN} --version
  if [[ $? -eq 0 ]]
  then
    echo ""
    echo "pylint available, proceeding with pylint sanity check."
    echo ""
  else
    echo ""
    echo "pylint not available."
    echo ""
    return 1
  fi

  # Configure pylint using the following file
  PYLINTRC_FILE="${SCRIPT_DIR}/pylintrc"

  if [[ ! -f "${PYLINTRC_FILE}" ]]; then
    die "ERROR: Cannot find pylint rc file at ${PYLINTRC_FILE}"
  fi

  # Run pylint in parallel, after some disk setup
  NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)
  NUM_CPUS=$(num_cpus)

  echo "Running pylint on ${NUM_SRC_FILES} files with ${NUM_CPUS} "\
"parallel jobs..."
  echo ""

  PYLINT_START_TIME=$(date +'%s')
  OUTPUT_FILE="$(mktemp)_pylint_output.log"

  rm -rf ${OUTPUT_FILE}

  # When running, filter to only contain the error code lines. Removes module
  # header, removes lines of context that show up from some lines.
  # Also, don't redirect stderr as this would hide pylint fatal errors.
  ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
      --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} | grep '\[[CEFW]' > ${OUTPUT_FILE}
  PYLINT_END_TIME=$(date +'%s')

  echo ""
  echo "pylint took $((PYLINT_END_TIME - PYLINT_START_TIME)) s"
  echo ""

  # Determine counts of errors
  N_ERRORS=$(wc -l ${OUTPUT_FILE} | cut -d' ' -f1)

  echo ""
  if [[ ${N_ERRORS} != 0 ]]; then
    echo "FAIL: Found ${N_ERRORS} errors"
    echo "Please correct these. If they must be ignored, use '# pylint: disable=<error name>' comments."
    cat ${OUTPUT_FILE}
    return 1
  else
    echo "PASS: No error found"
    return 0
  fi
}

# Run pep8 check
do_pep8() {
  # Usage: do_pep8 [--incremental]
  # Options:
  #   --incremental  Performs check on only the python files changed in the
  #                  last non-merge git commit.

  PEP8_BIN="/usr/local/bin/pep8"
  PEP8_CONFIG_FILE="${SCRIPT_DIR}/pep8"

  if [[ "$1" == "--incremental" ]]; then
    PYTHON_SRC_FILES=$(get_py_files_to_check --incremental)
    NUM_PYTHON_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)

    echo "do_pep8 will perform checks on only the ${NUM_PYTHON_SRC_FILES} "\
"Python file(s) changed in the last non-merge git commit due to the "\
"--incremental flag:"
    echo "${PYTHON_SRC_FILES}"
    echo ""
  else
    PYTHON_SRC_FILES=$(get_py_files_to_check)
  fi

  if [[ -z ${PYTHON_SRC_FILES} ]]; then
    echo "do_pep8 found no Python files to check. Returning."
    return 0
  fi

  if [[ ! -f "${PEP8_CONFIG_FILE}" ]]; then
    die "ERROR: Cannot find pep8 config file at ${PEP8_CONFIG_FILE}"
  fi
  echo "See \"${PEP8_CONFIG_FILE}\" for pep8 config( e.g., ignored errors)"

  NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)

  echo "Running pep8 on ${NUM_SRC_FILES} files"
  echo ""

  PEP8_START_TIME=$(date +'%s')
  PEP8_OUTPUT_FILE="$(mktemp)_pep8_output.log"

  rm -rf ${PEP8_OUTPUT_FILE}

  ${PEP8_BIN} --config="${PEP8_CONFIG_FILE}" --statistics \
      ${PYTHON_SRC_FILES} 2>&1 | tee ${PEP8_OUTPUT_FILE}
  PEP8_END_TIME=$(date +'%s')

  echo ""
  echo "pep8 took $((PEP8_END_TIME - PEP8_START_TIME)) s"
  echo ""

  if [[ -s ${PEP8_OUTPUT_FILE} ]]; then
    echo "FAIL: pep8 found above errors and/or warnings."
    return 1
  else
    echo "PASS: No pep8 errors or warnings were found"
    return 0
  fi
}


do_buildifier(){
  BUILD_FILES=$(find tensorflow -name 'BUILD*')
  NUM_BUILD_FILES=$(echo ${BUILD_FILES} | wc -w)

  echo "Running do_buildifier on ${NUM_BUILD_FILES} files"
  echo ""

  BUILDIFIER_START_TIME=$(date +'%s')
  BUILDIFIER_OUTPUT_FILE="$(mktemp)_buildifier_output.log"

  rm -rf ${BUILDIFIER_OUTPUT_FILE}

  buildifier -v -mode=check \
    ${BUILD_FILES} 2>&1 | tee ${BUILDIFIER_OUTPUT_FILE}
  BUILDIFIER_END_TIME=$(date +'%s')

  echo ""
  echo "buildifier took $((BUILDIFIER_END_TIME - BUILDIFIER_START_TIME)) s"
  echo ""

  if [[ -s ${BUILDIFIER_OUTPUT_FILE} ]]; then
    echo "FAIL: buildifier found errors and/or warnings in above BUILD files."
    echo "buildifier suggested the following changes:"
    buildifier -v -mode=diff -diff_command=diff ${BUILD_FILES}
    echo "Please fix manually or run buildifier <file> to auto-fix."
    return 1
  else
    echo "PASS: No buildifier errors or warnings were found"
    return 0
  fi
}

do_external_licenses_check(){
  BUILD_TARGET="$1"
  LICENSES_TARGET="$2"

  EXTERNAL_LICENSES_CHECK_START_TIME=$(date +'%s')

  EXTERNAL_DEPENDENCIES_FILE="$(mktemp)_external_dependencies.log"
  LICENSES_FILE="$(mktemp)_licenses.log"
  MISSING_LICENSES_FILE="$(mktemp)_missing_licenses.log"
  EXTRA_LICENSES_FILE="$(mktemp)_extra_licenses.log"
  TMP_FILE="$(mktemp)_tmp.log"

  echo "Getting external dependencies for ${BUILD_TARGET}"
 bazel cquery "attr('licenses', 'notice', deps(${BUILD_TARGET}))" --keep_going > "${TMP_FILE}" 2>&1
 cat "${TMP_FILE}" \
  | grep -e "^\/\/" -e "^@" \
  | grep -E -v "^//tensorflow" \
  | sed -e 's|:.*||' \
  | sort \
  | uniq 2>&1 \
  | tee ${EXTERNAL_DEPENDENCIES_FILE}

  echo
  echo "Getting list of external licenses mentioned in ${LICENSES_TARGET}."
  bazel cquery "deps(${LICENSES_TARGET})" --keep_going > "${TMP_FILE}" 2>&1
 cat "${TMP_FILE}" \
  | grep -e "^\/\/" -e "^@" \
  | grep -E -v "^//tensorflow" \
  | sed -e 's|:.*||' \
  | sort \
  | uniq 2>&1 \
  | tee ${LICENSES_FILE}

  echo
  comm -1 -3 ${EXTERNAL_DEPENDENCIES_FILE}  ${LICENSES_FILE} 2>&1 | tee ${EXTRA_LICENSES_FILE}
  echo
  comm -2 -3 ${EXTERNAL_DEPENDENCIES_FILE}  ${LICENSES_FILE} 2>&1 | tee ${MISSING_LICENSES_FILE}

  EXTERNAL_LICENSES_CHECK_END_TIME=$(date +'%s')

  # Denylist
  echo ${MISSING_LICENSES_FILE}
  grep \
    -e "@bazel_tools//platforms" \
    -e "@bazel_tools//third_party/" \
    -e "@bazel_tools//tools" \
    -e "@local" \
    -e "@com_google_absl//absl" \
    -e "@org_tensorflow//" \
    -e "@com_github_googlecloudplatform_google_cloud_cpp//google" \
    -e "@com_github_grpc_grpc//src/compiler" \
    -e "@platforms//os" \
    -e "@ruy//" \
    -v ${MISSING_LICENSES_FILE} > temp.txt
  mv temp.txt ${MISSING_LICENSES_FILE}

  # Allowlist
  echo ${EXTRA_LICENSE_FILE}
  grep \
    -e "//third_party/mkl" \
    -e "//third_party/mkl_dnn" \
    -e "@bazel_tools//src" \
    -e "@bazel_tools//platforms" \
    -e "@bazel_tools//tools/" \
    -e "@org_tensorflow//tensorflow" \
    -e "@com_google_absl//" \
    -e "//external" \
    -e "@local" \
    -e "@com_github_googlecloudplatform_google_cloud_cpp//" \
    -e "@embedded_jdk//" \
    -e "^//$" \
    -e "@ruy//" \
    -v ${EXTRA_LICENSES_FILE} > temp.txt
  mv temp.txt ${EXTRA_LICENSES_FILE}



  echo
  echo "do_external_licenses_check took $((EXTERNAL_LICENSES_CHECK_END_TIME - EXTERNAL_LICENSES_CHECK_START_TIME)) s"
  echo

  if [[ -s ${MISSING_LICENSES_FILE} ]] || [[ -s ${EXTRA_LICENSES_FILE} ]] ; then
    echo "FAIL: mismatch in packaged licenses and external dependencies"
    if [[ -s ${MISSING_LICENSES_FILE} ]] ; then
      echo "Missing the licenses for the following external dependencies:"
      cat ${MISSING_LICENSES_FILE}
      echo "Please add the license(s) to ${LICENSES_TARGET}."
    fi
    if [[ -s ${EXTRA_LICENSES_FILE} ]] ; then
      echo "Please remove the licenses for the following external dependencies from target ${LICENSES_TARGET}."
      cat ${EXTRA_LICENSES_FILE}
    fi
    rm -rf ${EXTERNAL_DEPENDENCIES_FILE}
    rm -rf ${LICENSES_FILE}
    rm -rf ${MISSING_LICENSES_FILE}
    rm -rf ${EXTRA_LICENSES_FILE}
    return 1
  else
    echo "PASS: all external licenses included."
    rm -rf ${EXTERNAL_DEPENDENCIES_FILE}
    rm -rf ${LICENSES_FILE}
    rm -rf ${MISSING_LICENSES_FILE}
    rm -rf ${EXTRA_LICENSES_FILE}
    return 0
  fi
}

do_pip_package_licenses_check() {
  echo "Running do_pip_package_licenses_check"
  echo ""
  do_external_licenses_check \
    "//tensorflow/tools/pip_package:build_pip_package" \
    "//tensorflow/tools/pip_package:licenses"
}

do_lib_package_licenses_check() {
  echo "Running do_lib_package_licenses_check"
  echo ""
  do_external_licenses_check \
    "//tensorflow:libtensorflow.so" \
    "//tensorflow/tools/lib_package:clicenses_generate"
}

do_java_package_licenses_check() {
  echo "Running do_java_package_licenses_check"
  echo ""
  do_external_licenses_check \
    "//tensorflow/java:libtensorflow_jni.so" \
    "//tensorflow/tools/lib_package:jnilicenses_generate"
}

#Check for the bazel cmd status (First arg is error message)
cmd_status(){
  if [[ $? != 0 ]]; then
    echo ""
    echo "FAIL: ${BUILD_CMD}"
    echo "  $1 See lines above for details."
    return 1
  else
    echo ""
    echo "PASS: ${BUILD_CMD}"
    return 0
  fi
}

# Run bazel build --nobuild to test the validity of the BUILD files
do_bazel_nobuild() {
  BUILD_TARGET="//tensorflow/..."
  BUILD_TARGET="${BUILD_TARGET} -//tensorflow/lite/..."
  BUILD_CMD="bazel build --nobuild ${BAZEL_FLAGS} -- ${BUILD_TARGET}"

  ${BUILD_CMD}

  cmd_status \
    "This is due to invalid BUILD files."
}

do_bazel_deps_query() {
  local BUILD_TARGET='//tensorflow/...'
  # Android targets tend to depend on an Android runtime being available.
  # Exclude until the sanity test has such a runtime available.
  #
  # TODO(mikecase): Remove TF Lite exclusion from this list. Exclusion is
  # necessary since the @androidsdk WORKSPACE dependency is commented out by
  # default in TF WORKSPACE file.
  local BUILD_TARGET="${BUILD_TARGET}"' - kind("android_*", //tensorflow/...)'

  # We've set the flag noimplicit_deps as a workaround for
  # https://github.com/bazelbuild/bazel/issues/10544
  bazel query ${BAZEL_FLAGS} --noimplicit_deps -- "deps($BUILD_TARGET)" > /dev/null

  cmd_status \
    "This is due to invalid BUILD files."
}

do_pip_smoke_test() {
  cd "$ROOT_DIR/tensorflow/tools/pip_package"
  python pip_smoke_test.py
}

do_code_link_check() {
  tensorflow/tools/ci_build/code_link_check.sh
}

# List .h|.cc files changed in the last non-merge git commit that still exist,
# i.e., not removed.
# Usage: get_clang_files_to_check [--incremental]
get_clang_files_to_check() {
  if [[ "$1" == "--incremental" ]]; then
    CHANGED_CLANG_FILES=$(get_changed_files_in_last_non_merge_git_commit | \
                       grep '.*\.h$\|.*\.cc$')

    # Do not include files removed in the last non-merge commit.
    CLANG_FILES=""
    for CLANG_FILE in ${CHANGED_CLANG_FILES}; do
      if [[ -f "${CLANG_FILE}" ]]; then
        CLANG_FILES="${CLANG_FILES} ${CLANG_FILE}"
      fi
    done

    echo "${CLANG_FILES}"
  else
    find tensorflow -name '*.h' -o -name '*.cc'
  fi
}

do_clang_format_check() {
  if [[ $# != "0" ]] && [[ $# != "1" ]]; then
    echo "Invalid syntax when invoking do_clang_format_check"
    echo "Usage: do_clang_format_check [--incremental]"
    return 1
  fi

  if [[ "$1" == "--incremental" ]]; then
    CLANG_SRC_FILES=$(get_clang_files_to_check --incremental)

    if [[ -z "${CLANG_SRC_FILES}" ]]; then
      echo "do_clang_format_check will NOT run due to --incremental flag and "\
"due to the absence of .h or .cc code changes in the last commit."
      return 0
    fi
  elif [[ -z "$1" ]]; then
    # TODO (yongtang): Always pass --incremental until all files have
    # been sanitized gradually. Then this --incremental could be removed.
    CLANG_SRC_FILES=$(get_clang_files_to_check --incremental)
  else
    echo "Invalid syntax for invoking do_clang_format_check"
    echo "Usage: do_clang_format_check [--incremental]"
    return 1
  fi

  CLANG_FORMAT=${CLANG_FORMAT:-clang-format-3.8}

  success=1
  for filename in $CLANG_SRC_FILES; do
    $CLANG_FORMAT --style=google $filename | diff $filename - > /dev/null
    if [ ! $? -eq 0 ]; then
      success=0
      echo File $filename is not properly formatted with "clang-format "\
"--style=google"
    fi
  done

  if [ $success == 0 ]; then
    echo Clang format check fails.
    exit 1
  fi
  echo Clang format check success.
}

do_check_load_py_test() {
  cd "$ROOT_DIR/tensorflow/tools/pip_package"
  python check_load_py_test.py
}

do_check_file_name_test() {
  cd "$ROOT_DIR/tensorflow/tools/test"
  python file_name_test.py
}

# Check that TARGET does not depend on DISALLOWED_DEP.
_check_no_deps() {
  TARGET="$1"
  DISALLOWED_DEP="$2"
  EXTRA_FLAG="$3"

  TMP_FILE="$(mktemp)_tmp.log"
  echo "Checking ${TARGET} does not depend on ${DISALLOWED_DEP} ..."
  bazel cquery ${EXTRA_FLAG} "somepath(${TARGET}, ${DISALLOWED_DEP})" --keep_going> "${TMP_FILE}" 2>&1
  if cat "${TMP_FILE}" | grep "Empty query results"; then
      echo "Success."
  else
      cat "${TMP_FILE}"
      echo
      echo "ERROR: Found path from ${TARGET} to disallowed dependency ${DISALLOWED_DEP}."
      echo "See above for path."
      rm "${TMP_FILE}"
      exit 1
  fi
  rm "${TMP_FILE}"
}

_do_pip_no_cuda_deps_check() {
  EXTRA_FLAG="$1"
  DISALLOWED_CUDA_DEPS=("@local_config_cuda//cuda:cudart"
        "@local_config_cuda//cuda:cublas"
        "@local_config_cuda//cuda:cuda_driver"
        "@local_config_cuda//cuda:cudnn"
        "@local_config_cuda//cuda:curand"
        "@local_config_cuda//cuda:cusolver"
        "@local_config_cuda//cuda:cusparse"
        "@local_config_tensorrt//:tensorrt")
  for cuda_dep in "${DISALLOWED_CUDA_DEPS[@]}"
  do
   _check_no_deps "//tensorflow/tools/pip_package:build_pip_package" "${cuda_dep}" "${EXTRA_FLAG}"
   RESULT=$?

   if [[ ${RESULT} != "0" ]]; then
    exit 1
   fi
  done
}

do_pip_no_cuda_deps_check_ubuntu() {
  _do_pip_no_cuda_deps_check "--@local_config_cuda//:enable_cuda"
}

do_pip_no_cuda_deps_check_windows() {
  _do_pip_no_cuda_deps_check "--@local_config_cuda//:enable_cuda --define framework_shared_object=false"
}

do_configure_test() {
  for WITH_CUDA in 1 0
  do
    export TF_NEED_CUDA=${WITH_CUDA}
    export CUDNN_INSTALL_PATH="/usr/local/cudnn"
    export PYTHON_BIN_PATH=$(which python3.8)
    yes "" | ${PYTHON_BIN_PATH} configure.py

    RESULT=$?
    if [[ ${RESULT} != "0" ]]; then
     exit 1
    fi
  done
}

# Supply all sanity step commands and descriptions
SANITY_STEPS=("do_configure_test" "do_buildifier" "do_bazel_nobuild" "do_bazel_deps_query" "do_pip_package_licenses_check" "do_lib_package_licenses_check" "do_java_package_licenses_check" "do_pip_smoke_test" "do_check_load_py_test" "do_code_link_check" "do_check_file_name_test" "do_pip_no_cuda_deps_check_ubuntu" "do_pip_no_cuda_deps_check_windows")
SANITY_STEPS_DESC=("Run ./configure" "buildifier check" "bazel nobuild" "bazel query" "pip: license check for external dependencies" "C library: license check for external dependencies" "Java Native Library: license check for external dependencies" "Pip Smoke Test: Checking py_test dependencies exist in pip package" "Check load py_test: Check that BUILD files with py_test target properly load py_test" "Code Link Check: Check there are no broken links" "Check file names for cases" "Check Ubuntu gpu pip package does not depend on cuda shared libraries" "Check Windows gpu pip package does not depend on cuda shared libraries")
INCREMENTAL_FLAG=""
DEFAULT_BAZEL_CONFIGS=""

# Parse command-line arguments
BAZEL_FLAGS=${DEFAULT_BAZEL_CONFIGS}
for arg in "$@"; do
  if [[ "${arg}" == "--pep8" ]]; then
    # Only run pep8 test if "--pep8" option supplied
    SANITY_STEPS=("do_pep8")
    SANITY_STEPS_DESC=("pep8 test")
  elif [[ "${arg}" == "--incremental" ]]; then
    INCREMENTAL_FLAG="--incremental"
  elif [[ "${arg}" == "--pylint" ]]; then
    SANITY_STEPS=("do_pylint")
    SANITY_STEPS_DESC=("pylint test")
  else
    BAZEL_FLAGS="${BAZEL_FLAGS} ${arg}"
  fi
done


FAIL_COUNTER=0
PASS_COUNTER=0
STEP_EXIT_CODES=()

# Execute all the sanity build steps
COUNTER=0
while [[ ${COUNTER} -lt "${#SANITY_STEPS[@]}" ]]; do
  INDEX=COUNTER
  ((INDEX++))

  echo ""
  echo "=== Sanity check step ${INDEX} of ${#SANITY_STEPS[@]}: "\
"${SANITY_STEPS[COUNTER]} (${SANITY_STEPS_DESC[COUNTER]}) ==="
  echo ""

  # subshell: don't leak variables or changes of working directory
  (
  ${SANITY_STEPS[COUNTER]} ${INCREMENTAL_FLAG}
  )
  RESULT=$?

  if [[ ${RESULT} != "0" ]]; then
    ((FAIL_COUNTER++))
  else
    ((PASS_COUNTER++))
  fi

  STEP_EXIT_CODES+=(${RESULT})

  echo ""
  ((COUNTER++))
done

# Print summary of build results
COUNTER=0
echo "==== Summary of sanity check results ===="
TESTCASE_XML=''
while [[ ${COUNTER} -lt "${#SANITY_STEPS[@]}" ]]; do
  INDEX=COUNTER
  ((INDEX++))

  echo "${INDEX}. ${SANITY_STEPS[COUNTER]}: ${SANITY_STEPS_DESC[COUNTER]}"
  TESTCASE_XML="${TESTCASE_XML} <testcase name=\"${SANITY_STEPS_DESC[COUNTER]}\" status=\"run\" classname=\"\" time=\"0\">"

  if [[ ${STEP_EXIT_CODES[COUNTER]} == "0" ]]; then
    printf "  ${COLOR_GREEN}PASS${COLOR_NC}\n"
  else
    printf "  ${COLOR_RED}FAIL${COLOR_NC}\n"
    TESTCASE_XML="${TESTCASE_XML} <failure message=\"\" type=\"\"/>"
  fi

  TESTCASE_XML="${TESTCASE_XML} </testcase>"

  ((COUNTER++))
done

echo
echo "${FAIL_COUNTER} failed; ${PASS_COUNTER} passed."

mkdir -p "${KOKORO_ARTIFACTS_DIR}/${KOKORO_JOB_NAME}/summary"
echo '<?xml version="1.0" encoding="UTF-8"?>'\
  '<testsuites name="1"  tests="1" failures="0" errors="0" time="0">'\
  '<testsuite name="Kokoro Summary" tests="'"$((FAIL_COUNTER + PASS_COUNTER))"\
  '" failures="'"${FAIL_COUNTER}"'" errors="0" time="0">'\
  "${TESTCASE_XML}"'</testsuite></testsuites>'\
  > "${KOKORO_ARTIFACTS_DIR}/${KOKORO_JOB_NAME}/summary/sponge_log.xml"

echo
if [[ ${FAIL_COUNTER} == "0" ]]; then
  printf "Sanity checks ${COLOR_GREEN}PASSED${COLOR_NC}\n"
else
  printf "Sanity checks ${COLOR_RED}FAILED${COLOR_NC}\n"
  exit 1
fi
