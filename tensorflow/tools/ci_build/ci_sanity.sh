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
# Usage: ci_sanity.sh [options]
#
# Options:
#           run sanity checks: python 2&3 pylint checks and bazel nobuild
#  --pep8   run pep8 test only
#  --incremental  Performs checks incrementally, by using the files changed in
#                 the latest commit

# Current script directory
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )

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
  echo $(git rev-list --no-merges -n 1 HEAD)
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
    echo $(find tensorflow -name '*.py')
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
"^tensorflow/python/platform/default/_googletest\.py.*\[E0102.*function already defined "\
"^tensorflow/python/platform/gfile\.py.*\[E0301.*non-iterator"

  echo "ERROR_WHITELIST=\"${ERROR_WHITELIST}\""

  if [[ $# != "1" ]] && [[ $# != "2" ]]; then
    echo "Invalid syntax when invoking do_pylint"
    echo "Usage: do_pylint (PYTHON2 | PYTHON3) [--incremental]"
    return 1
  fi

  if [[ $1 == "PYTHON2" ]]; then
    PYLINT_BIN="python /usr/local/lib/python2.7/dist-packages/pylint/lint.py"
  elif [[ $1 == "PYTHON3" ]]; then
    PYLINT_BIN="python3 /usr/local/lib/python3.4/dist-packages/pylint/lint.py"
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
  rm -rf ${ERRORS_FLIE}
  rm -rf ${NONWL_ERRORS_FILE}
  touch ${NONWL_ERRORS_FILE}

  ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
      --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} 2>&1 > ${OUTPUT_FILE}
  PYLINT_END_TIME=$(date +'%s')

  echo ""
  echo "pylint took $((${PYLINT_END_TIME} - ${PYLINT_START_TIME})) s"
  echo ""

  grep -E '(\[E|\[W0311|\[W0312)' ${OUTPUT_FILE} > ${ERRORS_FILE}

  N_ERRORS=0
  while read LINE; do
    IS_WHITELISTED=0
    for WL_REGEX in ${ERROR_WHITELIST}; do
      if [[ ! -z $(echo ${LINE} | grep "${WL_REGEX}") ]]; then
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
  echo "pep8 took $((${PEP8_END_TIME} - ${PEP8_START_TIME})) s"
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
  echo "buildifier took $((${BUILDIFIER_END_TIME} - ${BUILDIFIER_START_TIME})) s"
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

# Run bazel build --nobuild to test the validity of the BUILD files
do_bazel_nobuild() {
  BUILD_TARGET="//tensorflow/..."
  BUILD_CMD="bazel build --nobuild ${BUILD_TARGET}"

  ${BUILD_CMD}

  if [[ $? != 0 ]]; then
    echo ""
    echo "FAIL: ${BUILD_CMD}"
    echo "  This is due to invalid BUILD files. See lines above for details."
    return 1
  else
    echo ""
    echo "PASS: ${BUILD_CMD}"
    return 0
  fi
}

# Supply all sanity step commands and descriptions
SANITY_STEPS=("do_pylint PYTHON2" "do_pylint PYTHON3" "do_buildifier" "do_bazel_nobuild")
SANITY_STEPS_DESC=("Python 2 pylint" "Python 3 pylint" "buildifier check" "bazel nobuild")

INCREMENTAL_FLAG=""

# Parse command-line arguments
for arg in "$@"; do
  if [[ "${arg}" == "--pep8" ]]; then
    # Only run pep8 test if "--pep8" option supplied
    SANITY_STEPS=("do_pep8")
    SANITY_STEPS_DESC=("pep8 test")
  elif [[ "${arg}" == "--incremental" ]]; then
    INCREMENTAL_FLAG="--incremental"
  else
    echo "ERROR: Unrecognized command-line flag: $1"
    exit 1
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

  ${SANITY_STEPS[COUNTER]} ${INCREMENTAL_FLAG}
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
    echo "  PASS"
  else
    echo "  FAIL"
  fi

  ((COUNTER++))
done

echo ""
echo "${FAIL_COUNTER} failed; ${PASS_COUNTER} passed."

echo ""
if [[ ${FAIL_COUNTER} == "0" ]]; then
  echo "Sanity checks PASSED"
else
  die "Sanity checks FAILED"
fi
