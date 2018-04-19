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
  N_CPUS=$(grep -c ^processor /proc/cpuinfo)
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
  # Usage: do_pylint (PYTHON2 | PYTHON3) [--incremental]
  #
  # Options:
  #   --incremental  Performs check on only the python files changed in the
  #                  last non-merge git commit.

  # Use this list to whitelist pylint errors
  ERROR_WHITELIST="^tensorflow/python/framework/function_test\.py.*\[E1123.*noinline "\
"^tensorflow/python/platform/default/_gfile\.py.*\[E0301.*non-iterator "\
"^tensorflow/python/platform/default/_googletest\.py.*\[E0102.*function\salready\sdefined "\
"^tensorflow/python/feature_column/feature_column_test\.py.*\[E0110.*abstract-class-instantiated "\
"^tensorflow/contrib/layers/python/layers/feature_column\.py.*\[E0110.*abstract-class-instantiated "\
"^tensorflow/contrib/eager/python/evaluator\.py.*\[E0202.*method-hidden "\
"^tensorflow/contrib/eager/python/metrics_impl\.py.*\[E0202.*method-hidden "\
"^tensorflow/python/platform/gfile\.py.*\[E0301.*non-iterator "\
"^tensorflow/python/keras/_impl/keras/callbacks\.py.*\[E1133.*not-an-iterable "\
"^tensorflow/python/keras/_impl/keras/layers/recurrent\.py.*\[E0203.*access-member-before-definition "\
"^tensorflow/python/kernel_tests/constant_op_eager_test.py.*\[E0303.*invalid-length-returned"

  echo "ERROR_WHITELIST=\"${ERROR_WHITELIST}\""

  if [[ $# != "1" ]] && [[ $# != "2" ]]; then
    echo "Invalid syntax when invoking do_pylint"
    echo "Usage: do_pylint (PYTHON2 | PYTHON3) [--incremental]"
    return 1
  fi

  if [[ $1 == "PYTHON2" ]]; then
    PYLINT_BIN="python -m pylint"
  elif [[ $1 == "PYTHON3" ]]; then
    PYLINT_BIN="python3 -m pylint"
  else
    echo "Unrecognized python version (PYTHON2 | PYTHON3): $1"
    return 1
  fi

  if [[ "$2" == "--incremental" ]]; then
    PYTHON_SRC_FILES=$(get_py_files_to_check --incremental)

    if [[ -z "${PYTHON_SRC_FILES}" ]]; then
      echo "do_pylint will NOT run due to --incremental flag and due to the "\
"absence of Python code changes in the last commit."
      return 0
    else
      # For incremental builds, we still check all Python files in cases there
      # are function signature changes that affect unchanged Python files.
      PYTHON_SRC_FILES=$(get_py_files_to_check)
    fi
  elif [[ -z "$2" ]]; then
    PYTHON_SRC_FILES=$(get_py_files_to_check)
  else
    echo "Invalid syntax for invoking do_pylint"
    echo "Usage: do_pylint (PYTHON2 | PYTHON3) [--incremental]"
    return 1
  fi

  if [[ -z ${PYTHON_SRC_FILES} ]]; then
    echo "do_pylint found no Python files to check. Returning."
    return 0
  fi

  PYLINTRC_FILE="${SCRIPT_DIR}/pylintrc"

  if [[ ! -f "${PYLINTRC_FILE}" ]]; then
    die "ERROR: Cannot find pylint rc file at ${PYLINTRC_FILE}"
  fi

  NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)
  NUM_CPUS=$(num_cpus)

  echo "Running pylint on ${NUM_SRC_FILES} files with ${NUM_CPUS} "\
"parallel jobs..."
  echo ""

  PYLINT_START_TIME=$(date +'%s')
  OUTPUT_FILE="$(mktemp)_pylint_output.log"
  ERRORS_FILE="$(mktemp)_pylint_errors.log"
  NONWL_ERRORS_FILE="$(mktemp)_pylint_nonwl_errors.log"

  rm -rf ${OUTPUT_FILE}
  rm -rf ${ERRORS_FILE}
  rm -rf ${NONWL_ERRORS_FILE}
  touch ${NONWL_ERRORS_FILE}

  ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
      --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} > ${OUTPUT_FILE} 2>&1
  PYLINT_END_TIME=$(date +'%s')

  echo ""
  echo "pylint took $((PYLINT_END_TIME - PYLINT_START_TIME)) s"
  echo ""

  # Report only what we care about
  # Ref https://pylint.readthedocs.io/en/latest/technical_reference/features.html
  # E: all errors
  # W0311 bad-indentation
  # W0312 mixed-indentation
  # C0330 bad-continuation
  # C0301 line-too-long
  # C0326 bad-whitespace
  # W0611 unused-import
  # W0622 redefined-builtin
  grep -E '(\[E|\[W0311|\[W0312|\[C0330|\[C0301|\[C0326|\[W0611|\[W0622)' ${OUTPUT_FILE} > ${ERRORS_FILE}

  N_ERRORS=0
  while read -r LINE; do
    IS_WHITELISTED=0
    for WL_REGEX in ${ERROR_WHITELIST}; do
      if echo ${LINE} | grep -q "${WL_REGEX}"; then
        echo "Found a whitelisted error:"
        echo "  ${LINE}"
        IS_WHITELISTED=1
      fi
    done

    if [[ ${IS_WHITELISTED} == "0" ]]; then
      echo "${LINE}" >> ${NONWL_ERRORS_FILE}
      echo "" >> ${NONWL_ERRORS_FILE}
      ((N_ERRORS++))
    fi
  done <${ERRORS_FILE}

  echo ""
  if [[ ${N_ERRORS} != 0 ]]; then
    echo "FAIL: Found ${N_ERRORS} non-whitelited pylint errors:"
    cat "${NONWL_ERRORS_FILE}"
    return 1
  else
    echo "PASS: No non-whitelisted pylint errors were found."
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

  buildifier -showlog -v -mode=check \
    ${BUILD_FILES} 2>&1 | tee ${BUILDIFIER_OUTPUT_FILE}
  BUILDIFIER_END_TIME=$(date +'%s')

  echo ""
  echo "buildifier took $((BUILDIFIER_END_TIME - BUILDIFIER_START_TIME)) s"
  echo ""

  if [[ -s ${BUILDIFIER_OUTPUT_FILE} ]]; then
    echo "FAIL: buildifier found errors and/or warnings in above BUILD files."
    echo "buildifier suggested the following changes:"
    buildifier -showlog -v -mode=diff ${BUILD_FILES}
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

  echo "Getting external dependencies for ${BUILD_TARGET}"
 bazel query "attr('licenses', 'notice', deps(${BUILD_TARGET}))" --keep_going \
  | grep -E -v "^//tensorflow" \
  | sed -e 's|:.*||' \
  | sort \
  | uniq 2>&1 \
  | tee ${EXTERNAL_DEPENDENCIES_FILE}

  echo
  echo "Getting list of external licenses mentioned in ${LICENSES_TARGET}."
  bazel query "deps(${LICENSES_TARGET})" --keep_going \
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

  # Blacklist
  echo ${MISSING_LICENSES_FILE}
  grep -e "@bazel_tools//third_party/" -e "@com_google_absl//absl" -e "@org_tensorflow//" -v ${MISSING_LICENSES_FILE} > temp.txt
  mv temp.txt ${MISSING_LICENSES_FILE}

  # Whitelist
  echo ${EXTRA_LICENSE_FILE}
  grep -e "@bazel_tools//src" -e "@bazel_tools//tools/" -e "@com_google_absl//" -e "//external" -e "@local" -v ${EXTRA_LICENSES_FILE} > temp.txt
  mv temp.txt ${EXTRA_LICENSES_FILE}



  echo
  echo "do_external_licenses_check took $((EXTERNAL_LICENSES_CHECK_END_TIME - EXTERNAL_LICENSES_CHECK_START_TIME)) s"
  echo

  if [[ -s ${MISSING_LICENSES_FILE} ]] || [[ -s ${EXTRA_LICENSES_FILE} ]] ; then
    echo "FAIL: mismatch in packaged licenses and external dependencies"
    if [[ -s ${MISSING_LICENSES_FILE} ]] ; then
      echo "Missing the licenses for the following external dependencies:"
      cat ${MISSING_LICENSES_FILE}
    fi
    if [[ -s ${EXTRA_LICENSES_FILE} ]] ; then
      echo "Please remove the licenses for the following external dependencies:"
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
# TODO(mikecase): Remove TF Lite exclusion from this list. Exclusion is
# necessary since the @androidsdk WORKSPACE dependency is commented
# out by default in TF WORKSPACE file.
do_bazel_nobuild() {
  BUILD_TARGET="//tensorflow/..."
  BUILD_TARGET="${BUILD_TARGET} -//tensorflow/contrib/lite/java/demo/app/..."
  BUILD_TARGET="${BUILD_TARGET} -//tensorflow/contrib/lite/examples/android/..."
  BUILD_TARGET="${BUILD_TARGET} -//tensorflow/contrib/lite/schema/..."
  BUILD_CMD="bazel build --nobuild ${BAZEL_FLAGS} -- ${BUILD_TARGET}"

  ${BUILD_CMD}

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

do_cmake_python_sanity() {
  cd "$ROOT_DIR/tensorflow/contrib/cmake"
  python -m unittest -v python_sanity_test
}

do_check_futures_test() {
  cd "$ROOT_DIR/tensorflow/tools/test"
  python check_futures_test.py
}

do_check_file_name_test() {
  cd "$ROOT_DIR/tensorflow/tools/test"
  python file_name_test.py
}

# Supply all sanity step commands and descriptions
SANITY_STEPS=("do_pylint PYTHON2" "do_pylint PYTHON3" "do_check_futures_test" "do_buildifier" "do_bazel_nobuild" "do_pip_package_licenses_check" "do_lib_package_licenses_check" "do_java_package_licenses_check" "do_pip_smoke_test" "do_check_load_py_test" "do_code_link_check" "do_cmake_python_sanity" "do_check_file_name_test")
SANITY_STEPS_DESC=("Python 2 pylint" "Python 3 pylint" "Check that python files have certain __future__ imports" "buildifier check" "bazel nobuild" "pip: license check for external dependencies" "C library: license check for external dependencies" "Java Native Library: license check for external dependencies" "Pip Smoke Test: Checking py_test dependencies exist in pip package" "Check load py_test: Check that BUILD files with py_test target properly load py_test" "Code Link Check: Check there are no broken links" "Test entries in /tensorflow/contrib/cmake/python_{modules|protos|protos_cc}.txt for validity and consistency" "Check file names for cases")

INCREMENTAL_FLAG=""
DEFAULT_BAZEL_CONFIGS="--config=hdfs --config=gcp"

# Parse command-line arguments
BAZEL_FLAGS=${DEFAULT_BAZEL_CONFIGS}
for arg in "$@"; do
  if [[ "${arg}" == "--pep8" ]]; then
    # Only run pep8 test if "--pep8" option supplied
    SANITY_STEPS=("do_pep8")
    SANITY_STEPS_DESC=("pep8 test")
  elif [[ "${arg}" == "--incremental" ]]; then
    INCREMENTAL_FLAG="--incremental"
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
while [[ ${COUNTER} -lt "${#SANITY_STEPS[@]}" ]]; do
  INDEX=COUNTER
  ((INDEX++))

  echo "${INDEX}. ${SANITY_STEPS[COUNTER]}: ${SANITY_STEPS_DESC[COUNTER]}"
  if [[ ${STEP_EXIT_CODES[COUNTER]} == "0" ]]; then
    printf "  ${COLOR_GREEN}PASS${COLOR_NC}\n"
  else
    printf "  ${COLOR_RED}FAIL${COLOR_NC}\n"
  fi

  ((COUNTER++))
done

echo
echo "${FAIL_COUNTER} failed; ${PASS_COUNTER} passed."

echo
if [[ ${FAIL_COUNTER} == "0" ]]; then
  printf "Sanity checks ${COLOR_GREEN}PASSED${COLOR_NC}\n"
else
  printf "Sanity checks ${COLOR_RED}FAILED${COLOR_NC}\n"
  exit 1
fi
