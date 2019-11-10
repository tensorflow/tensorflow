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
# Build the Python PIP installation package for TensorFlow and install
# the package.
# The PIP installation is done using the --user flag.
#
# Usage:
#   pip.sh CONTAINER_TYPE [--test_tutorials] [--integration_tests] [bazel flags]
#
# When executing the Python unit tests, the script obeys the shell
# variables: TF_BUILD_BAZEL_CLEAN, TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES,
# NO_TEST_ON_INSTALL, PIP_TEST_ROOT, TF_NIGHTLY
#
# TF_BUILD_BAZEL_CLEAN, if set to any non-empty and non-0 value, directs the
# script to perform bazel clean prior to main build and test steps.
#
# TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES overrides the default extra pip packages
# to be installed in virtualenv before run_pip_tests.sh is called. Multiple
# pakcage names are separated with spaces.
#
# If NO_TEST_ON_INSTALL has any non-empty and non-0 value, the test-on-install
# part will be skipped.
#
# If NO_TEST_USER_OPS has any non-empty and non-0 value, the testing of user-
# defined ops against the installation will be skipped.
#
# If NO_TEST_TFDBG_BINARIES has any non-empty and non-0 value, the testing of
# TensorFlow Debugger (tfdbg) binaries and examples will be skipped.
#
# If PIP_TEST_ROOT has a non-empty and a non-0 value, the whl files will be
# placed in that directory.
#
# If TF_NIGHTLY has a non-empty and a non-0 value, the name of the project will
# be changed to tf_nightly or tf_nightly_gpu.
#
# Any flags not listed in the usage above will be passed directly to Bazel.
#
# If the --test_tutorials flag is set, it will cause the script to run the
# tutorial tests (see test_tutorials.sh) after the PIP
# installation and the Python unit tests-on-install step. Likewise,
# --integration_tests will cause the integration tests (integration_tests.sh)
# to run.
#

# Helper function: Strip leading and trailing whitespaces
str_strip () {
  echo -e "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Fixed naming patterns for wheel (.whl) files given different python versions
if [[ $(uname) == "Linux" ]]; then
  declare -A WHL_TAGS
  WHL_TAGS=(["2.7"]="cp27-none" ["3.4"]="cp34-cp34m" ["3.5"]="cp35-cp35m")
fi


INSTALL_EXTRA_PIP_PACKAGES=${TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES}


# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"


SKIP_RETURN_CODE=112


# Get the command line arguments
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift

if [[ -n "${TF_BUILD_BAZEL_CLEAN}" ]] && \
   [[ "${TF_BUILD_BAZEL_CLEAN}" != "0" ]]; then
  echo "TF_BUILD_BAZEL_CLEAN=${TF_BUILD_BAZEL_CLEAN}: Performing 'bazel clean'"
  bazel clean
fi

DO_TEST_USER_OPS=1
if [[ -n "${NO_TEST_USER_OPS}" ]] && \
   [[ "${NO_TEST_USER_OPS}" != "0" ]]; then
  echo "NO_TEST_USER_OPS=${NO_TEST_USER_OPS}: Will skip testing of user ops"
  DO_TEST_USER_OPS=0
fi

DO_TEST_TFDBG_BINARIES=1
if [[ -n "${NO_TEST_TFDBG_BINARIES}" ]] && \
   [[ "${NO_TEST_TFDBG_BINARIES}" != "0" ]]; then
  echo "NO_TEST_TFDBG_BINARIES=${NO_TEST_TFDBG_BINARIES}: Will skip testing of tfdbg binaries"
  DO_TEST_TFDBG_BINARIES=0
fi

DO_TEST_TUTORIALS=0
DO_INTEGRATION_TESTS=0
BAZEL_FLAGS=""
while true; do
  if [[ "${1}" == "--test_tutorials" ]]; then
    DO_TEST_TUTORIALS=1
  elif [[ "${1}" == "--integration_tests" ]]; then
    DO_INTEGRATION_TESTS=1
  else
    BAZEL_FLAGS="${BAZEL_FLAGS} ${1}"
  fi

  shift
  if [[ -z "${1}" ]]; then
    break
  fi
done

BAZEL_FLAGS=$(str_strip "${BAZEL_FLAGS}")

if [[ -z "$GIT_TAG_OVERRIDE" ]]; then
  BAZEL_FLAGS+=" --action_env=GIT_TAG_OVERRIDE"
fi

echo "Using Bazel flags: ${BAZEL_FLAGS}"

PIP_BUILD_TARGET="//tensorflow/tools/pip_package:build_pip_package"
GPU_FLAG=""
ROCM_FLAG=""
if [[ ${CONTAINER_TYPE} == "cpu" ]] || \
   [[ ${CONTAINER_TYPE} == "debian.jessie.cpu" ]]; then
  bazel build ${BAZEL_FLAGS} ${PIP_BUILD_TARGET} || \
      die "Build failed."
elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  bazel build ${BAZEL_FLAGS} ${PIP_BUILD_TARGET} || \
      die "Build failed."
  GPU_FLAG="--gpu"
elif [[ ${CONTAINER_TYPE} == "rocm" ]]; then
  bazel build ${BAZEL_FLAGS} ${PIP_BUILD_TARGET} || \
      die "Build failed."
  ROCM_FLAG="--rocm"
else
  die "Unrecognized container type: \"${CONTAINER_TYPE}\""
fi

MAC_FLAG=""
if [[ $(uname) == "Darwin" ]]; then
  MAC_FLAG="--mac"
fi


# Check if in a virtualenv
IN_VENV=$(python -c 'import sys; print("1" if hasattr(sys, "real_prefix") else "0")')
# If still in a virtualenv, deactivate it first
if [[ "$IN_VENV" == "1" ]]; then
  echo "It appears that we are already in a virtualenv. Deactivating..."
  deactivate || die "FAILED: Unable to deactivate from existing virtualenv"
fi

# Obtain the path to Python binary
source tools/python_bin_path.sh

# Assume: PYTHON_BIN_PATH is exported by the script above
if [[ -z "$PYTHON_BIN_PATH" ]]; then
  die "PYTHON_BIN_PATH was not provided. Did you run configure?"
fi

# Determine the major and minor versions of Python being used (e.g., 2.7)
# This info will be useful for determining the directory of the local pip
# installation of Python
PY_MAJOR_MINOR_VER=$(${PYTHON_BIN_PATH} -V 2>&1 | awk '{print $NF}' | cut -d. -f-2)
if [[ -z "${PY_MAJOR_MINOR_VER}" ]]; then
  die "ERROR: Unable to determine the major.minor version of Python"
fi

echo "Python binary path to be used in PIP install: ${PYTHON_BIN_PATH} "\
"(Major.Minor version: ${PY_MAJOR_MINOR_VER})"

# Create a TF_NIGHTLY argument if this is a nightly build
PROJECT_NAME="tensorflow"
NIGHTLY_FLAG=""
if [ -n "$TF_NIGHTLY" ]; then
  PROJECT_NAME="tf_nightly"
  NIGHTLY_FLAG="--nightly_flag"
fi

# Build PIP Wheel file
# Set default pip file folder unless specified by env variable
if [ -z "$PIP_TEST_ROOT" ]; then
  PIP_TEST_ROOT="pip_test"
fi
PIP_WHL_DIR="${PIP_TEST_ROOT}/whl"
PIP_WHL_DIR=$(realpath ${PIP_WHL_DIR})  # Get absolute path
rm -rf ${PIP_WHL_DIR} && mkdir -p ${PIP_WHL_DIR}
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} ${GPU_FLAG} ${ROCM_FLAG} ${NIGHTLY_FLAG} || \
    die "build_pip_package FAILED"

WHL_PATH=$(ls ${PIP_WHL_DIR}/${PROJECT_NAME}*.whl)
if [[ $(echo ${WHL_PATH} | wc -w) -ne 1 ]]; then
  die "ERROR: Failed to find exactly one built TensorFlow .whl file in "\
"directory: ${PIP_WHL_DIR}"
fi

# Print the size of the PIP wheel file.
echo
echo "Size of the PIP wheel file built: $(ls -l ${WHL_PATH} | awk '{print $5}')"
echo

# Rename the whl file properly so it will have the python
# version tags and platform tags that won't cause pip install issues.
if [[ $(uname) == "Linux" ]]; then
  PY_TAGS=${WHL_TAGS[${PY_MAJOR_MINOR_VER}]}
  PLATFORM_TAG=$(to_lower "$(uname)_$(uname -m)")
# MAC has bash v3, which does not have associative array
elif [[ $(uname) == "Darwin" ]]; then
  if [[ ${PY_MAJOR_MINOR_VER} == "2.7" ]]; then
    PY_TAGS="py2-none"
  elif [[ ${PY_MAJOR_MINOR_VER} == "3.5" ]]; then
    PY_TAGS="py3-none"
  elif [[ ${PY_MAJOR_MINOR_VER} == "3.6" ]]; then
    PY_TAGS="py3-none"
  fi
  PLATFORM_TAG="any"
fi

WHL_DIR=$(dirname "${WHL_PATH}")
WHL_BASE_NAME=$(basename "${WHL_PATH}")

if [[ -n "${PY_TAGS}" ]]; then
  NEW_WHL_BASE_NAME=$(echo ${WHL_BASE_NAME} | cut -d \- -f 1)-\
$(echo ${WHL_BASE_NAME} | cut -d \- -f 2)-${PY_TAGS}-${PLATFORM_TAG}.whl

  if [[ ! -f "${WHL_DIR}/${NEW_WHL_BASE_NAME}" ]]; then
    if cp "${WHL_DIR}/${WHL_BASE_NAME}" "${WHL_DIR}/${NEW_WHL_BASE_NAME}"
    then
      echo "Copied wheel file: ${WHL_BASE_NAME} --> ${NEW_WHL_BASE_NAME}"
    else
      die "ERROR: Failed to copy wheel file to ${NEW_WHL_BASE_NAME}"
    fi
  fi
fi

if [[ $(uname) == "Linux" ]]; then
  AUDITED_WHL_NAME="${WHL_DIR}/$(echo ${WHL_BASE_NAME//linux/manylinux1})"

  # Repair the wheels for cpu manylinux1
  if [[ ${CONTAINER_TYPE} == "cpu" ]]; then
    echo "auditwheel repairing ${WHL_PATH}"
    auditwheel repair -w ${WHL_DIR} ${WHL_PATH}

    if [[ -f ${AUDITED_WHL_NAME} ]]; then
      WHL_PATH=${AUDITED_WHL_NAME}
      echo "Repaired manylinx1 wheel file at: ${WHL_PATH}"
    else
      die "ERROR: Cannot find repaired wheel."
    fi
  # Copy and rename for gpu manylinux as we do not want auditwheel to package in libcudart.so
  elif [[ ${CONTAINER_TYPE} == "gpu" ]] || \
       [[ ${CONTAINER_TYPE} == "rocm" ]]; then
    WHL_PATH=${AUDITED_WHL_NAME}
    cp ${WHL_DIR}/${WHL_BASE_NAME} ${WHL_PATH}
    echo "Copied manylinx1 wheel file at ${WHL_PATH}"
  fi
fi


create_activate_virtualenv_and_install_tensorflow() {
  # Create and activate a virtualenv; then install tensorflow pip package in it.
  #
  # Usage:
  #   create_activate_virtualenv_and_install_tensorflow [--clean] \
  #       <VIRTUALENV_DIR> <TF_WHEEL_PATH>
  #
  # Arguments:
  #   --clean: Create a clean virtualenv, i.e., without --system-site-packages.
  #   VIRTUALENV_DIR: virtualenv directory to be created.
  #   TF_WHEEL_PATH: Path to the tensorflow wheel file to be installed in the
  #     virtualenv.

  VIRTUALENV_FLAGS="--system-site-packages"
  if [[ "$1" == "--clean" ]]; then
    VIRTUALENV_FLAGS=""
    shift
  fi

  VIRTUALENV_DIR="$1"
  TF_WHEEL_PATH="$2"
  if [[ -d "${VIRTUALENV_DIR}" ]]; then
    if rm -rf "${VIRTUALENV_DIR}"
    then
      echo "Removed existing virtualenv directory: ${VIRTUALENV_DIR}"
    else
      die "Failed to remove existing virtualenv directory: ${VIRTUALENV_DIR}"
    fi
  fi

  if mkdir -p "${VIRTUALENV_DIR}"
  then
    echo "Created virtualenv directory: ${VIRTUALENV_DIR}"
  else
    die "FAILED to create virtualenv directory: ${VIRTUALENV_DIR}"
  fi

  # Use the virtualenv from the default python version (i.e., python-virtualenv)
  # to create the virtualenv directory for testing. Use the -p flag to specify
  # the python version inside the to-be-created virtualenv directory.
  ${PYTHON_BIN_PATH} -m virtualenv -p "${PYTHON_BIN_PATH}" ${VIRTUALENV_FLAGS} \
    "${VIRTUALENV_DIR}" || \
    die "FAILED: Unable to create virtualenv"

  source "${VIRTUALENV_DIR}/bin/activate" || \
    die "FAILED: Unable to activate virtualenv in ${VIRTUALENV_DIR}"

  # Install the pip file in virtual env.

  # Upgrade pip so it supports tags such as cp27mu, manylinux1 etc.
  echo "Upgrade pip in virtualenv"

  # NOTE: pip install --upgrade pip leads to a documented TLS issue for
  # some versions in python
  curl https://bootstrap.pypa.io/get-pip.py | python

  # Force upgrade of setuptools. This must happen before the pip install of the
  # WHL_PATH, which pulls in absl-py, which uses install_requires notation
  # introduced in setuptools >=20.5. The default version of setuptools is 5.5.1,
  # which is too old for absl-py.
  pip install --upgrade setuptools==39.1.0

  # Force tensorflow reinstallation. Otherwise it may not get installed from
  # last build if it had the same version number as previous build.
  PIP_FLAGS="--upgrade --force-reinstall"
  pip install -v ${PIP_FLAGS} ${WHL_PATH} || \
    die "pip install (forcing to reinstall tensorflow) FAILED"
  echo "Successfully installed pip package ${TF_WHEEL_PATH}"

  # Force downgrade of setuptools. This must happen after the pip install of the
  # WHL_PATH, which ends up upgrading to the latest version of setuptools.
  # Versions of setuptools >= 39.1.0 will cause tests to fail like this:
  #   ImportError: cannot import name py31compat
  pip install --upgrade setuptools==39.1.0
}

################################################################################
# Smoke test of tensorflow install in clean virtualenv
################################################################################
do_clean_virtualenv_smoke_test() {
  if [[ -n "${NO_TEST_ON_INSTALL}" ]] &&
       [[ "${NO_TEST_ON_INSTALL}" != "0" ]]; then
    echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
    echo "  Skipping smoke test of tensorflow install in clean virtualenv"
    return ${SKIP_RETURN_CODE}
  fi

  CLEAN_VENV_DIR="${PIP_TEST_ROOT}/venv_clean"
  create_activate_virtualenv_and_install_tensorflow --clean \
    "${CLEAN_VENV_DIR}" "${WHL_PATH}"

  # cd to a temporary directory to avoid picking up Python files in the source
  # tree.
  TMP_DIR=$(mktemp -d)
  pushd "${TMP_DIR}"
  if [[ $(python -c "import tensorflow as tf; print(tf.Session().run(tf.constant(42)))") == 42 ]];
  then
    echo "Smoke test of tensorflow install in clean virtualenv PASSED."
  else
    echo "Smoke test of tensorflow install in clean virtualenv FAILED."
    return 1
  fi

  deactivate
  if [[ $? != 0 ]]; then
    echo "FAILED: Unable to deactivate virtualenv from ${CLEAN_VENV_DIR}"
    return 1
  fi

  popd
  rm -rf "${TMP_DIR}" "${CLEAN_VENV_DIR}"
}

################################################################################
# Perform installation of tensorflow in "non-clean" virtualenv and tests against
# the install.
################################################################################
do_virtualenv_pip_test() {
  # Create virtualenv directory for install test
  VENV_DIR="${PIP_TEST_ROOT}/venv"
  create_activate_virtualenv_and_install_tensorflow \
    "${VENV_DIR}" "${WHL_PATH}"

  # Install extra pip packages required by the test-on-install
  for PACKAGE in ${INSTALL_EXTRA_PIP_PACKAGES}; do
    echo "Installing extra pip package required by test-on-install: ${PACKAGE}"

    pip install ${PACKAGE}
    if [[ $? != 0 ]]; then
      echo "pip install ${PACKAGE} FAILED"
      return 1
    fi
  done

  if [[ -n "${NO_TEST_ON_INSTALL}" ]] &&
     [[ "${NO_TEST_ON_INSTALL}" != "0" ]]; then
    echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
    echo "  Skipping ALL Python unit tests on install"
    return ${SKIP_RETURN_CODE}
  else
    # Call run_pip_tests.sh to perform test-on-install
    "${SCRIPT_DIR}/run_pip_tests.sh" --virtualenv ${GPU_FLAG} ${ROCM_FLAG} ${MAC_FLAG}
    if [[ $? != 0 ]]; then
      echo "PIP tests-on-install FAILED"
      return 1
    fi
  fi
}

################################################################################
# Run tests tagged with oss_serial against the virtualenv install.
################################################################################
do_virtualenv_oss_serial_pip_test() {
  if [[ -n "${NO_TEST_ON_INSTALL}" ]] &&
     [[ "${NO_TEST_ON_INSTALL}" != "0" ]]; then
    echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
    echo "  Skipping Python unit tests on install tagged with oss_serial"
    return ${SKIP_RETURN_CODE}
  else
    # Call run_pip_tests.sh to perform test-on-install
    "${SCRIPT_DIR}/run_pip_tests.sh" \
      --virtualenv ${GPU_FLAG} ${ROCM_FLAG} ${MAC_FLAG} --oss_serial
    if [[ $? != 0 ]]; then
      echo "PIP tests-on-install (oss_serial) FAILED"
      return 1
    fi
  fi
}

################################################################################
# Test user ops (optional).
################################################################################
do_test_user_ops() {
  if [[ "${DO_TEST_USER_OPS}" == "1" ]]; then
    "${SCRIPT_DIR}/test_user_ops.sh" --virtualenv ${GPU_FLAG} ${ROCM_FLAG}
    if [[ $? != 0 ]]; then
      echo "PIP user-op tests-on-install FAILED"
      return 1
    fi
  else
    echo "Skipping user-op test-on-install due to DO_TEST_USER_OPS = ${DO_TEST_USER_OPS}"
    return ${SKIP_RETURN_CODE}
  fi
}

################################################################################
# Test TensorFlow Debugger (tfdbg) binaries (optional).
################################################################################
do_test_tfdbg_binaries() {
  if [[ "${DO_TEST_TFDBG_BINARIES}" == "1" ]]; then
    # cd to a temporary directory to avoid picking up Python files in the source
    # tree.
    TMP_DIR=$(mktemp -d)
    pushd "${TMP_DIR}"

    "${SCRIPT_DIR}/../../../python/debug/examples/examples_test.sh" \
      --virtualenv
    if  [[ $? != 0 ]]; then
      echo "PIP tests-on-install of tfdbg binaries FAILED"
      return 1
    fi
    popd
  else
    echo "Skipping test of tfdbg binaries due to DO_TEST_TFDBG_BINARIES = ${DO_TEST_TFDBG_BINARIES}"
    return ${SKIP_RETURN_CODE}
  fi
}

################################################################################
# Test tutorials (optional).
################################################################################
do_test_tutorials() {
  if [[ "${DO_TEST_TUTORIALS}" == "1" ]]; then
    "${SCRIPT_DIR}/test_tutorials.sh" --virtualenv
    if [[ $? != 0 ]]; then
      echo "PIP tutorial tests-on-install FAILED"
      return 1
    fi
  else
    echo "Skipping tutorial tests-on-install due to DO_TEST_TUTORIALS = ${DO_TEST_TUTORIALS}"
    return ${SKIP_RETURN_CODE}
  fi
}

################################################################################
# Integration test for ffmpeg (optional).
################################################################################
do_ffmpeg_integration_test() {
  # Optional: Run integration tests
  if [[ "${DO_INTEGRATION_TESTS}" == "1" ]]; then
    "${SCRIPT_DIR}/integration_tests.sh" --virtualenv
    if [[ $? != 0 ]]; then
      echo "Integration tests on install FAILED"
      return 1
    fi
  else
    echo "Skipping ffmpeg integration due to DO_INTEGRATION_TESTS = ${DO_INTEGRATION_TESTS}"
    return ${SKIP_RETURN_CODE}
  fi
}


# List of all PIP test tasks and their descriptions.
PIP_TASKS=("do_clean_virtualenv_smoke_test" "do_virtualenv_pip_test" "do_virtualenv_oss_serial_pip_test" "do_test_user_ops" "do_test_tfdbg_binaries" "do_test_tutorials" "do_ffmpeg_integration_test")
PIP_TASKS_DESC=("Smoke test of pip install in clean virtualenv" "PIP tests in virtualenv" "PIP test in virtualenv (tag: oss_serial)" "User ops test" "TensorFlow Debugger (tfdbg) binaries test" "Tutorials test" "ffmpeg integration test")


# Execute all the PIP test steps.
COUNTER=0
FAIL_COUNTER=0
PASS_COUNTER=0
SKIP_COUNTER=0
while [[ ${COUNTER} -lt "${#PIP_TASKS[@]}" ]]; do
  INDEX=COUNTER
  ((INDEX++))

  echo
  printf "${COLOR_BOLD}=== PIP test step ${INDEX} of ${#PIP_TASKS[@]}: "\
"${PIP_TASKS[COUNTER]} (${PIP_TASKS_DESC[COUNTER]}) ===${COLOR_NC}"
  echo

  ${PIP_TASKS[COUNTER]}
  RESULT=$?

  if [[ ${RESULT} == ${SKIP_RETURN_CODE} ]]; then
    ((SKIP_COUNTER++))
  elif [[ ${RESULT} != "0" ]]; then
    ((FAIL_COUNTER++))
  else
    ((PASS_COUNTER++))
  fi

  STEP_EXIT_CODES+=(${RESULT})

  echo ""
  ((COUNTER++))
done

deactivate || die "FAILED: Unable to deactivate virtualenv from ${VENV_DIR}"


# Print summary of build results
COUNTER=0
echo "==== Summary of PIP test results ===="
while [[ ${COUNTER} -lt "${#PIP_TASKS[@]}" ]]; do
  INDEX=COUNTER
  ((INDEX++))

  echo "${INDEX}. ${PIP_TASKS[COUNTER]}: ${PIP_TASKS_DESC[COUNTER]}"
  if [[ ${STEP_EXIT_CODES[COUNTER]} == ${SKIP_RETURN_CODE} ]]; then
    printf "  ${COLOR_LIGHT_GRAY}SKIP${COLOR_NC}\n"
  elif [[ ${STEP_EXIT_CODES[COUNTER]} == "0" ]]; then
    printf "  ${COLOR_GREEN}PASS${COLOR_NC}\n"
  else
    printf "  ${COLOR_RED}FAIL${COLOR_NC}\n"
  fi

  ((COUNTER++))
done

echo
echo "${SKIP_COUNTER} skipped; ${FAIL_COUNTER} failed; ${PASS_COUNTER} passed."

echo
if [[ ${FAIL_COUNTER} == "0" ]]; then
  printf "PIP test ${COLOR_GREEN}PASSED${COLOR_NC}\n"
else
  printf "PIP test ${COLOR_RED}FAILED${COLOR_NC}\n"
  exit 1
fi
