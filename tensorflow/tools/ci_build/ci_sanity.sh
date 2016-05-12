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


# Subfunctions for substeps
# Run pylint
do_pylint() {
  # Use this list to whitelist pylint errors
  ERROR_WHITELIST="^tensorflow/python/framework/function_test\.py.*\[E1123.*noinline "\
"^tensorflow/python/platform/default/_gfile\.py.*\[E0301.*non-iterator "\
"^tensorflow/python/platform/default/_googletest\.py.*\[E0102.*function already defined "\
"^tensorflow/python/platform/gfile\.py.*\[E0301.*non-iterator"

  echo "ERROR_WHITELIST=\"${ERROR_WHITELIST}\""

  PYTHON_SRC_FILES=$(find tensorflow -name '*.py')

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

  pylint --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
      --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} 2>&1 > ${OUTPUT_FILE}
  PYLINT_END_TIME=$(date +'%s')

  echo ""
  echo "pylint took $((${PYLINT_END_TIME} - ${PYLINT_START_TIME})) s"
  echo ""

  grep '\[E' ${OUTPUT_FILE} > ${ERRORS_FILE}

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


FAIL=0

# Pylint
do_pylint || FAIL=1

# bazel nobuild
do_bazel_nobuild || FAIL=1

echo ""
if [[ ${FAIL} == "0" ]]; then
  echo "Sanity checks PASSED"
else
  die "Sanity checks FAILED"
fi
