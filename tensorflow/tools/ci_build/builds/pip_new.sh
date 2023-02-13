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
# Build the Python PIP installation package for TensorFlow and install
# the package.
#
# Usage:
#   pip_new.sh
#
# Required step(s):
#   Run configure.py prior to running this script.
#
# Required environment variable(s):
#   CONTAINER_TYPE:      (CPU | GPU)
#   OS_TYPE:             (UBUNTU | MACOS)
#   TF_PYTHON_VERSION:   ( python3.6 | python3.7 | python3.8 )
#   TF_BUILD_FLAGS:      Bazel build flags.
#                          e.g. TF_BUILD_FLAGS="--config=opt"
#   TF_TEST_FLAGS:       Bazel test flags.
#                          e.g. TF_TEST_FLAGS="--verbose_failures=true \
#                               --build_tests_only --test_output=errors"
#   TF_TEST_FILTER_TAGS: Filtering tags for bazel tests. More specifically,
#                        input tags for `--test_filter_tags` flag.
#                          e.g. TF_TEST_FILTER_TAGS="no_pip,-nomac,no_oss"
#   TF_TEST_TARGETS:     Bazel test targets.
#                          e.g. TF_TEST_TARGETS="//tensorflow/... \
#                               -//tensorflow/contrib/... \
#                               -//tensorflow/python/..."
#   IS_NIGHTLY:          Nightly run flag.
#                          e.g. IS_NIGHTLY=1  # nightly runs
#                          e.g. IS_NIGHTLY=0  # non-nightly runs
#
# Optional environment variables. If provided, overwrites any default values.
#   TF_PIP_TESTS:        PIP tests to run. If NOT specified, skips all tests.
#                          e.g. TF_PIP_TESTS="test_pip_virtualenv_clean \
#                               test_pip_virtualenv_clean \
#                               test_pip_virtualenv_oss_serial"
#   TF_PROJECT_NAME:     Name of the project. This string will be pass onto
#                        the wheel file name. For nightly builds, it will be
#                        overwritten to 'tf_nightly'. For gpu builds, '_gpu'
#                        will be appended.
#                          e.g. TF_PROJECT_NAME="tensorflow"
#                          e.g. TF_PROJECT_NAME="tf_nightly_gpu"
#   TF_PIP_TEST_ROOT:    Root directory for building and testing pip pkgs.
#                          e.g. TF_PIP_TEST_ROOT="pip_test"
#   TF_BUILD_BOTH_GPU_PACKAGES:    (1 | 0)
#                                  1 will build both tensorflow (w/ gpu support)
#                                  and tensorflow-gpu pip package. Will
#                                  automatically handle adding/removing of _gpu
#                                  suffix depending on what project name was
#                                  passed. Only work for Ubuntu.
#   TF_BUILD_BOTH_CPU_PACKAGES:    (1 | 0)
#                                  1 will build both tensorflow (no gpu support)
#                                  and tensorflow-cpu pip package. Will
#                                  automatically handle adding/removing of _cpu
#                                  suffix depending on what project name was
#                                  passed. Only work for MacOS
#   AUDITWHEEL_TARGET_PLAT:    Manylinux platform tag that is to be used for
#                              tagging the linux wheel files. By default, it is
#                              set to `manylinux2010` . For manylinux2014
#                              builds, change to `manylinux2014`.
#
# To-be-deprecated variable(s).
#   GIT_TAG_OVERRIDE:    Values for `--git_tag_override`. This flag gets passed
#                        in as `--action_env` for bazel build and tests.
#   TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES:
#                        Additional pip packages to be installed.
#                        Caveat: pip version needs to be checked prior.
#
# ==============================================================================

# set bash options
set -e
set -x

###########################################################################
# General helper function(s)
###########################################################################

# Strip leading and trailing whitespaces
str_strip () {
  echo -e "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Convert string to all lower case
lowercase() {
  if [[ -z "${1}" ]]; then
    die "Nothing to convert to lowercase. No argument given."
  fi
  echo "${1}" | tr '[:upper:]' '[:lower:]'
}

check_global_vars() {
  # Check container type
  if ! [[ ${CONTAINER_TYPE} == "cpu" ]] && \
     ! [[ ${CONTAINER_TYPE} == "rocm" ]] && \
     ! [[ ${CONTAINER_TYPE} == "gpu" ]]; then
    die "Error: Provided CONTAINER_TYPE \"${CONTAINER_TYPE}\" "\
        "is not supported."
  fi
  # Check OS type
  if ! [[ ${OS_TYPE} == "ubuntu" ]] && \
     ! [[ ${OS_TYPE} == "macos" ]]; then
    die "Error: Provided OS_TYPE \"${OS_TYPE}\" is not supported."
  fi
  # Check build flags
  if [[ -z ${TF_BUILD_FLAGS} ]]; then
    die "Error: TF_BUILD_FLAGS is not specified."
  fi
  # Check test flags
  if [[ -z ${TF_TEST_FLAGS} ]]; then
    die "Error: TF_TEST_FLAGS is not specified."
  fi
  # Check test filter tags
  if [[ -z ${TF_TEST_FILTER_TAGS} ]]; then
    die "Error: TF_TEST_FILTER_TAGS is not specified."
  fi
  # Check test targets
  if [[ -z ${TF_TEST_TARGETS} ]]; then
    die "Error: TF_TEST_TARGETS is not specified."
  fi
  # Check nightly status
  if [[ -z ${IS_NIGHTLY} ]]; then
    die "Error: IS_NIGHTLY is not specified."
  fi
}

add_test_filter_tag() {
  EMPTY=""
  while true; do
    FILTER="${1:$EMPTY}"
    if ! [[ $TF_TEST_FILTER_TAGS == *"${FILTER}"* ]]; then
      TF_TEST_FILTER_TAGS="${FILTER},${TF_TEST_FILTER_TAGS}"
    fi
    shift
    if [[ -z "${1}" ]]; then
      break
    fi
  done
}

remove_test_filter_tag() {
  EMPTY=""
  while true; do
    FILTER="${1:$EMPTY}"
    TF_TEST_FILTER_TAGS="$(echo ${TF_TEST_FILTER_TAGS} | sed -e 's/^'${FILTER}',//g' -e 's/,'${FILTER}'//g')"
    shift
    if [[ -z "${1}" ]]; then
      break
    fi
  done
}

# Clean up bazel build & test flags with proper configuration.
update_bazel_flags() {
  # Add git tag override flag if necessary.
  GIT_TAG_STR=" --action_env=GIT_TAG_OVERRIDE"
  if [[ -z "${GIT_TAG_OVERRIDE}" ]] && \
    ! [[ ${TF_BUILD_FLAGS} = *${GIT_TAG_STR}* ]]; then
    TF_BUILD_FLAGS+="${GIT_TAG_STR}"
  fi
  # Clean up whitespaces
  TF_BUILD_FLAGS=$(str_strip "${TF_BUILD_FLAGS}")
  TF_TEST_FLAGS=$(str_strip "${TF_TEST_FLAGS}")
  # Cleaned bazel flags
  echo "Bazel build flags (cleaned):\n" "${TF_BUILD_FLAGS}"
  echo "Bazel test flags (cleaned):\n" "${TF_TEST_FLAGS}"
}

update_test_filter_tags() {
  # Add test filter tags
  # This script is for validating built PIP packages. Add pip tags.
  add_test_filter_tag -no_pip -nopip
  # MacOS filter tags
  if [[ ${OS_TYPE} == "macos" ]]; then
    remove_test_filter_tag nomac no_mac
    add_test_filter_tag -nomac -no_mac
  fi
  echo "Final test filter tags: ${TF_TEST_FILTER_TAGS}"
}

# Check currently running python and pip version
check_python_pip_version() {
  # Check if only the major version of python is provided by the user.
  MAJOR_VER_ONLY=0
  if [[ ${#PYTHON_VER} -lt 9 ]]; then
    # User only provided major version (e.g. 'python3' instead of 'python3.7')
    MAJOR_VER_ONLY=1
  fi

  # Retrieve only the version number of the user requested python.
  PYTHON_VER_REQUESTED=${PYTHON_VER:6:3}
  echo "PYTHON_VER_REQUESTED: ${PYTHON_VER_REQUESTED}"

  # Retrieve only the version numbers of the python & pip in use currently.
  PYTHON_VER_IN_USE=$(python --version 2>&1)
  PYTHON_VER_IN_USE=${PYTHON_VER_IN_USE:7:3}
  PIP_VER_IN_USE=$(${PIP_BIN_PATH} --version)
  PIP_VER_IN_USE=${PIP_VER_IN_USE:${#PIP_VER_IN_USE}-4:3}

  # If only major versions are applied, drop minor versions.
  if [[ $MAJOR_VER_ONLY == 1 ]]; then
    PYTHON_VER_IN_USE=${PYTHON_VER_IN_USE:0:1}
    PIP_VER_IN_USE=${PIP_VER_IN_USE:0:1}
  fi

  # Check if all versions match.
  echo -e "User requested python version: '${PYTHON_VER_REQUESTED}'\n" \
    "Detected python version in use: '${PYTHON_VER_IN_USE}'\n"\
    "Detected pip version in use: '${PIP_VER_IN_USE}'"
  if ! [[ $PYTHON_VER_REQUESTED == $PYTHON_VER_IN_USE ]]; then
    die "Error: Mismatch in python versions detected."
  else:
    echo "Python and PIP versions in use match the requested."
  fi
}

# Write an entry to the sponge key-value store for this job.
write_to_sponge() {
  # The location of the key-value CSV file sponge imports.
  TF_SPONGE_CSV="${KOKORO_ARTIFACTS_DIR}/custom_sponge_config.csv"
  echo "$1","$2" >> "${TF_SPONGE_CSV}"
}


###########################################################################
# Setup: directories, local/global variables
###########################################################################

# Script directory and source necessary files.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"

# Required global variables
# Checks on values for these vars are done in "Build TF PIP Package" section.
CONTAINER_TYPE=$(lowercase "${CONTAINER_TYPE}")
OS_TYPE=$(lowercase "${OS_TYPE}")
PYTHON_VER=$(lowercase "${TF_PYTHON_VERSION}")

# Python bin path
if [[ -z "$PYTHON_BIN_PATH" ]]; then
  die "Error: PYTHON_BIN_PATH was not provided. Did you run configure?"
fi

# Set optional environment variables; set to default in case not user defined.
DEFAULT_PIP_TESTS="" # Do not run any tests by default
DEFAULT_PROJECT_NAME="tensorflow"
DEFAULT_PIP_TEST_ROOT="pip_test"
DEFAULT_BUILD_BOTH_GPU_PACKAGES=0
DEFAULT_BUILD_BOTH_CPU_PACKAGES=0
DEFAULT_AUDITWHEEL_TARGET_PLAT="manylinux2010"
# Take in optional global variables
PIP_TESTS=${TF_PIP_TESTS:-$DEFAULT_PIP_TESTS}
PROJECT_NAME=${TF_PROJECT_NAME:-$DEFAULT_PROJECT_NAME}
PIP_TEST_ROOT=${TF_PIP_TEST_ROOT:-$DEFAULT_PIP_TEST_ROOT}
BUILD_BOTH_GPU_PACKAGES=${TF_BUILD_BOTH_GPU_PACKAGES:-$DEFAULT_BUILD_BOTH_GPU_PACKAGES}
BUILD_BOTH_CPU_PACKAGES=${TF_BUILD_BOTH_CPU_PACKAGES:-$DEFAULT_BUILD_BOTH_CPU_PACKAGES}
AUDITWHEEL_TARGET_PLAT=${TF_AUDITWHEEL_TARGET_PLAT:-$DEFAULT_AUDITWHEEL_TARGET_PLAT}

# Override breaking change in setuptools v60 (https://github.com/pypa/setuptools/pull/2896)
export SETUPTOOLS_USE_DISTUTILS=stdlib

# Local variables
PIP_WHL_DIR="${KOKORO_ARTIFACTS_DIR}/tensorflow/${PIP_TEST_ROOT}/whl"
mkdir -p "${PIP_WHL_DIR}"
PIP_WHL_DIR=$(realpath "${PIP_WHL_DIR}") # Get absolute path
WHL_PATH=""
# Determine the major.minor versions of python being used (e.g., 3.7).
# Useful for determining the directory of the local pip installation.
PY_MAJOR_MINOR_VER=$(${PYTHON_BIN_PATH} -c "print(__import__('sys').version)" 2>&1 | awk '{ print $1 }' | head -n 1 | cut -d. -f1-2)

if [[ -z "${PY_MAJOR_MINOR_VER}" ]]; then
  die "ERROR: Unable to determine the major.minor version of Python."
fi
echo "Python binary path to be used in PIP install: ${PYTHON_BIN_PATH} "\
"(Major.Minor version: ${PY_MAJOR_MINOR_VER})"
PYTHON_BIN_PATH_INIT=${PYTHON_BIN_PATH}
PIP_BIN_PATH="$(which pip${PY_MAJOR_MINOR_VER})"

# PIP packages
INSTALL_EXTRA_PIP_PACKAGES="h5py portpicker scipy scikit-learn ${TF_BUILD_INSTALL_EXTRA_PIP_PACKAGES}"

###########################################################################
# Build TF PIP Package
###########################################################################

# First remove any already existing binaries for a clean start and test.
if [[ -d ${PIP_TEST_ROOT} ]]; then
  echo "Test root directory ${PIP_TEST_ROOT} already exists. Deleting it."
  sudo rm -rf ${PIP_TEST_ROOT}
fi

# Check that global variables are properly set.
check_global_vars

# Check if in a virtualenv and exit if yes.
# TODO(rameshsampath): Python 3.10 has pip conflicts when using global env, so build in virtualenv
# Once confirmed to work, run builds for all python env in a virtualenv
if [[ "x${PY_MAJOR_MINOR_VER}x" != "x3.10x" ]]; then
  IN_VENV=$(python -c 'import sys; print("1" if sys.version_info.major == 3 and sys.prefix != sys.base_prefix else "0")')
  if [[ "$IN_VENV" == "1" ]]; then
    echo "It appears that we are already in a virtualenv. Deactivating..."
    deactivate || source deactivate || die "FAILED: Unable to deactivate from existing virtualenv."
  fi
fi

# Obtain the path to python binary as written by ./configure if it was run.
if [[ -e tools/python_bin_path.sh ]]; then
  source tools/python_bin_path.sh
fi
# Assume PYTHON_BIN_PATH is exported by the script above or the caller.
if [[ -z "$PYTHON_BIN_PATH" ]]; then
  die "PYTHON_BIN_PATH was not provided. Did you run configure?"
fi

if [[ "$IS_NIGHTLY" == 1 ]]; then
  ${PYTHON_BIN_PATH} -m pip install tb-nightly
else
  ${PYTHON_BIN_PATH} -m pip install tensorboard
fi

if [[ "x${PY_MAJOR_MINOR_VER}x" == "x3.8x" ]]; then
  ${PYTHON_BIN_PATH} -m pip uninstall -y protobuf
  ${PYTHON_BIN_PATH} -m pip install "protobuf < 4"
fi

# Bazel build the file.
PIP_BUILD_TARGET="//tensorflow/tools/pip_package:build_pip_package"
# Clean bazel cache.
bazel clean
# Clean up and update bazel flags
update_bazel_flags
# Build. This outputs the file `build_pip_package`.
bazel build \
  --action_env=PYTHON_BIN_PATH=${PYTHON_BIN_PATH} \
  ${TF_BUILD_FLAGS} \
  ${PIP_BUILD_TARGET} \
  || die "Error: Bazel build failed for target: '${PIP_BUILD_TARGET}'"

###########################################################################
# Test function(s)
###########################################################################

test_pip_virtualenv() {
  # Get args
  WHL_PATH=$1
  shift
  VENV_DIR_NAME=$1
  shift
  TEST_TYPE_FLAG=$1

  # Check test type args
  if ! [[ ${TEST_TYPE_FLAG} == "--oss_serial" ]] && \
     ! [[ ${TEST_TYPE_FLAG} == "--clean" ]] && \
     ! [[ ${TEST_TYPE_FLAG} == "" ]]; then
     die "Error: Wrong test type given. TEST_TYPE_FLAG=${TEST_TYPE_FLAG}"
  fi

  # Create virtualenv directory for test
  VENV_DIR="${PIP_TEST_ROOT}/${VENV_DIR_NAME}"

  # Activate virtualenv
  create_activate_virtualenv ${TEST_TYPE_FLAG} ${VENV_DIR}
  # Install TF with pip
  TIME_START=$SECONDS
  install_tensorflow_pip "${WHL_PATH}"
  TIME_ELAPSED=$(($SECONDS - $TIME_START))
  echo "Time elapsed installing tensorflow = ${TIME_ELAPSED} seconds"

  # cd to a temporary directory to avoid picking up Python files in the source
  # tree.
  TMP_DIR=$(mktemp -d)
  pushd "${TMP_DIR}"

  # Run a quick check on tensorflow installation.
  RET_VAL=$(python -c "import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)")

  # Return to original directory. Remove temp dirs.
  popd
  sudo rm -rf "${TMP_DIR}"

  # Check result to see if tensorflow is properly installed.
  if ! [[ ${RET_VAL} == *'(4,)'* ]]; then
    echo "PIP test on virtualenv (non-clean) FAILED"
    return 1
  fi

  # Install extra pip packages, if specified.
  for PACKAGE in ${INSTALL_EXTRA_PIP_PACKAGES}; do
    echo "Installing extra pip package required by test-on-install: ${PACKAGE}"

    ${PIP_BIN_PATH} install ${PACKAGE}
    if [[ $? != 0 ]]; then
      echo "${PIP_BIN_PATH} install ${PACKAGE} FAILED."
      deactivate || source deactivate || die "FAILED: Unable to deactivate from existing virtualenv."
      return 1
    fi
  done

  # Run bazel test.
  run_test_with_bazel ${TEST_TYPE_FLAG}
  RESULT=$?

  # Deactivate from virtualenv.
  deactivate || source deactivate || die "FAILED: Unable to deactivate from existing virtualenv."
  sudo rm -rf "${VENV_DIR}"

  return $RESULT
}

###########################################################################
# Test helper function(s)
###########################################################################

create_activate_virtualenv() {
  VIRTUALENV_FLAGS="--system-site-packages"
  if [[ "${1}" == "--clean" ]]; then
    VIRTUALENV_FLAGS=""
    shift
  elif [[ "${1}" == "--oss_serial" ]]; then
    shift
  fi

  VIRTUALENV_DIR="${1}"
  if [[ -d "${VIRTUALENV_DIR}" ]]; then
    if sudo rm -rf "${VIRTUALENV_DIR}"
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
  ${PYTHON_BIN_PATH_INIT} -m virtualenv -p ${PYTHON_BIN_PATH_INIT} ${VIRTUALENV_FLAGS} ${VIRTUALENV_DIR} || \
    ${PYTHON_BIN_PATH_INIT} -m venv ${VIRTUALENV_DIR} || \
    die "FAILED: Unable to create virtualenv"

  source "${VIRTUALENV_DIR}/bin/activate" || \
    die "FAILED: Unable to activate virtualenv in ${VIRTUALENV_DIR}"

  # Update .tf_configure.bazelrc with venv python path for bazel test.
  PYTHON_BIN_PATH="$(which python)"
  yes "" | ./configure
}

install_tensorflow_pip() {
  if [[ -z "${1}" ]]; then
    die "Please provide a proper wheel file path."
  fi

  # Set path to pip.
  PIP_BIN_PATH="${PYTHON_BIN_PATH} -m pip"

  # Print python and pip bin paths
  echo "PYTHON_BIN_PATH to be used to install the .whl: ${PYTHON_BIN_PATH}"
  echo "PIP_BIN_PATH to be used to install the .whl: ${PIP_BIN_PATH}"

  # Upgrade pip so it supports tags such as cp27mu, manylinux2010 etc.
  echo "Upgrade pip in virtualenv"

  # NOTE: pip install --upgrade pip leads to a documented TLS issue for
  # some versions in python
  curl https://bootstrap.pypa.io/get-pip.py | ${PYTHON_BIN_PATH} || \
    die "Error: pip install (get-pip.py) FAILED"

  # Check that requested python version matches configured one.
  check_python_pip_version

  # setuptools v60.0.0 introduced a breaking change on how distutils is linked
  # https://github.com/pypa/setuptools/blob/main/CHANGES.rst#v6000
  ${PIP_BIN_PATH} install --upgrade "setuptools" || \
    die "Error: setuptools install, upgrade FAILED"

  # Force tensorflow reinstallation. Otherwise it may not get installed from
  # last build if it had the same version number as previous build.
  PIP_FLAGS="--upgrade --force-reinstall"
  ${PIP_BIN_PATH} install ${PIP_FLAGS} ${WHL_PATH} || \
    die "pip install (forcing to reinstall tensorflow) FAILED"
  echo "Successfully installed pip package ${WHL_PATH}"

  # Install the future package in the virtualenv. Installing it in user system
  # packages does not appear to port it over when creating a virtualenv.
  #   ImportError: No module named builtins
  ${PIP_BIN_PATH} install --upgrade "future>=0.17.1" || \
    die "Error: future install, upgrade FAILED"

  # Install the gast package in the virtualenv. Installing it in user system
  # packages does not appear to port it over when creating a virtualenv.
  ${PIP_BIN_PATH} install --upgrade "gast==0.4.0" || \
    die "Error: gast install, upgrade FAILED"

}

run_test_with_bazel() {
  IS_OSS_SERIAL=0
  if [[ "${1}" == "--oss_serial" ]]; then
    IS_OSS_SERIAL=1
  fi
  TF_GPU_COUNT=${TF_GPU_COUNT:-4}

  # PIP tests should have a "different" path. Different than the one we place
  # virtualenv, because we are deleting and recreating it here.
  PIP_TEST_PREFIX=bazel_pip
  TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
  sudo rm -rf $TEST_ROOT
  mkdir -p $TEST_ROOT
  ln -s $(pwd)/tensorflow $TEST_ROOT/tensorflow

  if [[ "${IS_OSS_SERIAL}" == "1" ]]; then
    remove_test_filter_tag -no_oss
    remove_test_filter_tag -oss_serial
    add_test_filter_tag oss_serial
  else
    add_test_filter_tag -oss_serial
  fi

  # Clean the bazel cache
  bazel clean
  # Clean up flags before running bazel commands
  update_bazel_flags
  # Clean up and update test filter tags
  update_test_filter_tags

  # Figure out how many concurrent tests we can run and do run the tests.
  BAZEL_PARALLEL_TEST_FLAGS=""
  if [[ $CONTAINER_TYPE == "gpu" ]] || [[ $CONTAINER_TYPE == "rocm" ]]; then
    # Number of test threads is the number of GPU cards available.
    if [[ $OS_TYPE == "macos" ]]; then
      BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=1"
    else
      BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=${TF_GPU_COUNT} \
        --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute"
    fi
  else
    # Number of test threads is the number of physical CPUs.
    if [[ $OS_TYPE == "macos" ]]; then
      BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=$(sysctl -n hw.ncpu)"
    else
      BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=$(grep -c ^processor /proc/cpuinfo)"
    fi
  fi

  if [[ ${IS_OSS_SERIAL} == 1 ]]; then
    BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=1"
  fi

  TEST_TARGETS_SYMLINK=""
  for TARGET in ${TF_TEST_TARGETS[@]}; do
    TARGET_NEW=$(echo ${TARGET} | sed -e "s/\/\//\/\/${PIP_TEST_PREFIX}\//g")
    TEST_TARGETS_SYMLINK+="${TARGET_NEW} "
  done
  echo "Test targets (symlink): ${TEST_TARGETS_SYMLINK}"

  # Run the test.
  bazel test --build_tests_only ${TF_TEST_FLAGS} ${BAZEL_PARALLEL_TEST_FLAGS} --test_tag_filters=${TF_TEST_FILTER_TAGS} -k -- ${TEST_TARGETS_SYMLINK}

  unlink ${TEST_ROOT}/tensorflow
}

run_all_tests() {
  WHL_PATH=$1

  if [[ -z "${PIP_TESTS}" ]]; then
    echo "No test was specified to run. Skipping all tests."
    return 0
  fi
  FAIL_COUNTER=0
  PASS_COUNTER=0
  for TEST in ${PIP_TESTS[@]}; do

    # Run tests.
    case "${TEST}" in
    "test_pip_virtualenv_clean")
      test_pip_virtualenv ${WHL_PATH} venv_clean --clean
      ;;
    "test_pip_virtualenv_non_clean")
      test_pip_virtualenv ${WHL_PATH} venv
      ;;
    "test_pip_virtualenv_oss_serial")
      test_pip_virtualenv ${WHL_PATH} venv_oss --oss_serial
      ;;
    *)
      die "No matching test ${TEST} was found. Stopping test."
      ;;
    esac

    # Check and update the results.
    RETVAL=$?

    # Update results counter
    if [ ${RETVAL} -eq 0 ]; then
      echo "Test (${TEST}) PASSED. (PASS COUNTER: ${PASS_COUNTER})"
      PASS_COUNTER=$(($PASS_COUNTER+1))
    else
      echo "Test (${TEST}) FAILED. (FAIL COUNTER: ${FAIL_COUNTER})"
      FAIL_COUNTER=$(($FAIL_COUNTER+1))
    fi
  done
  printf "${PASS_COUNTER} PASSED | ${FAIL_COUNTER} FAILED"
  if [[ "${FAIL_COUNTER}" == "0" ]]; then
    printf "PIP tests ${COLOR_GREEN}PASSED${COLOR_NC}\n"
    return 0
  else:
    printf "PIP tests ${COLOR_RED}FAILED${COLOR_NC}\n"
    return 1
  fi
}

###########################################################################
# Build TF PIP Wheel file
###########################################################################

# Update the build flags for building whl.
# Flags: GPU, OS, tf_nightly, project name
GPU_FLAG=""
NIGHTLY_FLAG=""

# TF Nightly flag
if [[ "$IS_NIGHTLY" == 1 ]]; then
  # If 'nightly' is not specified in the project name already, then add.
  if ! [[ $PROJECT_NAME == *"nightly"* ]]; then
    echo "WARNING: IS_NIGHTLY=${IS_NIGHTLY} but requested project name \
    (PROJECT_NAME=${PROJECT_NAME}) does not include 'nightly' string. \
    Renaming it to 'tf_nightly'."
    PROJECT_NAME="tf_nightly"
  fi
  NIGHTLY_FLAG="--nightly_flag"
fi

# CPU / GPU flag
if [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  GPU_FLAG="--gpu"
  if ! [[ $PROJECT_NAME == *"gpu"* ]]; then
    # Only update PROJECT_NAME if TF_PROJECT_NAME is not set
    if [[ -z "${TF_PROJECT_NAME}" ]]; then
      echo "WARNING: GPU is specified but requested project name (PROJECT_NAME=${PROJECT_NAME}) \
      does not include 'gpu'. Appending '_gpu' to the project name."
      PROJECT_NAME="${PROJECT_NAME}_gpu"
    fi
  fi
fi

if [[ ${CONTAINER_TYPE} == "rocm" ]]; then
  GPU_FLAG="--rocm"
  if ! [[ $PROJECT_NAME == *"rocm"* ]]; then
    # Only update PROJECT_NAME if TF_PROJECT_NAME is not set
    if [[ -z "${TF_PROJECT_NAME}" ]]; then
      echo "WARNING: ROCM is specified but requested project name (PROJECT_NAME=${PROJECT_NAME}) \
      does not include 'rocm'. Appending '_rocm' to the project name."
      PROJECT_NAME="${PROJECT_NAME}_rocm"
    fi
  fi
fi

./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} ${GPU_FLAG} ${NIGHTLY_FLAG} "--project_name" ${PROJECT_NAME} || die "build_pip_package FAILED"

PY_DOTLESS_MAJOR_MINOR_VER=$(echo $PY_MAJOR_MINOR_VER | tr -d '.')
if [[ $PY_DOTLESS_MAJOR_MINOR_VER == "2" ]]; then
  PY_DOTLESS_MAJOR_MINOR_VER="27"
fi

# Set wheel path and verify that there is only one .whl file in the path.
WHL_PATH=$(ls "${PIP_WHL_DIR}"/"${PROJECT_NAME}"-*"${PY_DOTLESS_MAJOR_MINOR_VER}"*"${PY_DOTLESS_MAJOR_MINOR_VER}"*.whl)
if [[ $(echo "${WHL_PATH}" | wc -w) -ne 1 ]]; then
  echo "ERROR: Failed to find exactly one built TensorFlow .whl file in "\
  "directory: ${PIP_WHL_DIR}"
fi

WHL_DIR=$(dirname "${WHL_PATH}")

# Print the size of the wheel file and log to sponge.
WHL_SIZE=$(ls -l ${WHL_PATH} | awk '{print $5}')
echo "Size of the PIP wheel file built: ${WHL_SIZE}"
write_to_sponge TF_INFO_WHL_SIZE ${WHL_SIZE}

# Build the other GPU package.
if [[ "$BUILD_BOTH_GPU_PACKAGES" -eq "1" ]] || [[ "$BUILD_BOTH_CPU_PACKAGES" -eq "1" ]]; then

  if [[ "$BUILD_BOTH_GPU_PACKAGES" -eq "1" ]] && [[ "$BUILD_BOTH_CPU_PACKAGES" -eq "1" ]]; then
    die "ERROR: TF_BUILD_BOTH_GPU_PACKAGES and TF_BUILD_BOTH_GPU_PACKAGES cannot both be set. No additional package will be built."
  fi

  echo "====================================="
  if [[ "$BUILD_BOTH_GPU_PACKAGES" -eq "1" ]]; then
    if ! [[ ${OS_TYPE} == "ubuntu" ]]; then
      die "ERROR: pip_new.sh only support building both GPU wheels on ubuntu."
    fi
    echo "Building the other GPU pip package."
    PROJECT_SUFFIX="gpu"
  else
    if ! [[ ${OS_TYPE} == "macos" ]]; then
      die "ERROR: pip_new.sh only support building both CPU wheels on macos."
    fi
    echo "Building the other CPU pip package."
    PROJECT_SUFFIX="cpu"
  fi

  # Check container type
  if ! [[ ${CONTAINER_TYPE} == ${PROJECT_SUFFIX} ]]; then
    die "Error: CONTAINER_TYPE needs to be \"${PROJECT_SUFFIX}\" to build ${PROJECT_SUFFIX} packages. Got"\
        "\"${CONTAINER_TYPE}\" instead."
  fi
  if [[ "$PROJECT_NAME" == *_${PROJECT_SUFFIX} ]]; then
    NEW_PROJECT_NAME=${PROJECT_NAME}"_${PROJECT_SUFFIX}"
  else
    NEW_PROJECT_NAME="${PROJECT_NAME}_${PROJECT_SUFFIX}"
  fi
  echo "The given ${PROJECT_SUFFIX} \$PROJECT_NAME is ${PROJECT_NAME}. The additional ${PROJECT_SUFFIX}"\
  "pip package will have project name ${NEW_PROJECT_NAME}."

  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} ${GPU_FLAG} ${NIGHTLY_FLAG} "--project_name" ${NEW_PROJECT_NAME} || die "build_pip_package FAILED"
fi

# On MacOS we not have to rename the wheel because it is generated with the
# wrong tag.
if [[ ${OS_TYPE} == "macos" ]] ; then
  for WHL_PATH in $(ls ${PIP_WHL_DIR}/*macosx_10_15_x86_64.whl); do
    # change 10_15 to 10_14
    NEW_WHL_PATH=${WHL_PATH/macosx_10_15/macosx_10_14}
    mv ${WHL_PATH} ${NEW_WHL_PATH}
  done

  # Also change global WHL_PATH. Ignore above shadow and everywhere else
  NEW_WHL_PATH=${WHL_PATH/macosx_10_15/macosx_10_14}
  WHL_PATH=${NEW_WHL_PATH}
fi

# Run tests (if any is specified).
run_all_tests ${WHL_PATH}


if [[ ${OS_TYPE} == "ubuntu" ]] && \
   ! [[ ${CONTAINER_TYPE} == "rocm" ]] ; then
  # Avoid Python3.6 abnormality by installing auditwheel here.
  # TODO(rameshsampath) - Cleanup and remove the need for auditwheel install
  # Python 3.10 requires auditwheel > 2 and its already installed in common.sh
  if [[ $PY_MAJOR_MINOR_VER -ne "3.10" ]]; then
    set +e
    pip3 show auditwheel || "pip${PY_MAJOR_MINOR_VER}" show auditwheel
    # For tagging wheels as manylinux2014, auditwheel needs to >= 3.0.0
    pip3 install auditwheel==3.3.1 || "pip${PY_MAJOR_MINOR_VER}" install auditwheel==3.3.1
    sudo pip3 install auditwheel==3.3.1 || \
      sudo "pip${PY_MAJOR_MINOR_VER}" install auditwheel==3.3.1
    set -e
  fi
  auditwheel --version

  for WHL_PATH in $(ls ${PIP_WHL_DIR}/*.whl); do
    # Repair the wheels for cpu manylinux2010/manylinux2014
    echo "auditwheel repairing ${WHL_PATH}"
    auditwheel repair --plat ${AUDITWHEEL_TARGET_PLAT}_$(uname -m) -w "${WHL_DIR}" "${WHL_PATH}"

    if [[ $(ls ${WHL_DIR} | grep ${AUDITWHEEL_TARGET_PLAT} | wc -l) == 1 ]] ; then
      WHL_PATH=${WHL_DIR}/$(ls ${WHL_DIR} | grep ${AUDITWHEEL_TARGET_PLAT})
      echo "Repaired ${AUDITWHEEL_TARGET_PLAT} wheel file at: ${WHL_PATH}"
    else
      die "WARNING: Cannot find repaired wheel."
    fi
  done
fi
echo "EOF: Successfully ran pip_new.sh"
