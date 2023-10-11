#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# Shared functions to build and test Python package for TensorFlow on MacOS
# ==============================================================================

set -e
set -x

source tensorflow/tools/ci_build/release/common.sh

function die() {
  echo "$@" 1>&2 ; exit 1;
}

# Write an entry to the sponge key-value store for this job.
write_to_sponge() {
  # The location of the key-value CSV file sponge imports.
  TF_SPONGE_CSV="${KOKORO_ARTIFACTS_DIR}/custom_sponge_config.csv"
  echo "$1","$2" >> "${TF_SPONGE_CSV}"
}

# Runs bazel build and saves wheel files them in PIP_WHL_DIR
function bazel_build_wheel {

  if [[ -z "${1}" ]]; then
    die "Missing wheel file path to install and test build"
  fi
  PIP_WHL_DIR=$1
  shift
  PIP_WHL_FLAGS=$@

  mkdir -p "${PIP_WHL_DIR}"
  PIP_WHL_DIR=$(realpath "${PIP_WHL_DIR}") # Get absolute path

  VENV_DIR=".tf-venv"
  rm -rf "${VENV_DIR}"

  # Set up and install MacOS pip dependencies.
  python -m venv ${VENV_DIR} && source ${VENV_DIR}/bin/activate
  install_macos_pip_deps

  # Update .tf_configure.bazelrc with venv python path for bazel
  export PYTHON_BIN_PATH="$(which python)"
  yes "" | ./configure

  # Build the pip package
  bazel build \
    --config=release_cpu_macos \
    --action_env=PYENV_VERSION="${PYENV_VERSION}" \
    --action_env=PYTHON_BIN_PATH="${PYTHON_BIN_PATH}" \
    //tensorflow/tools/pip_package:build_pip_package \
    || die "Error: Bazel build failed"

  # Build the wheel
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} ${PIP_WHL_FLAGS}

  # Set wheel path and verify that there is only one .whl file in the path.
  WHL_PATH=$(ls "${PIP_WHL_DIR}"/*.whl)
  if [[ $(echo "${WHL_PATH}" | wc -w) -ne 1 ]]; then
    die "ERROR: Failed to find exactly one built TensorFlow .whl file in "\
    "directory: ${PIP_WHL_DIR}"
  fi

  # Print the size of the wheel file and log to sponge.
  WHL_SIZE=$(ls -l "${WHL_PATH}" | awk '{print $5}')
  echo "Size of the PIP wheel file built: ${WHL_SIZE}"
  write_to_sponge TF_INFO_WHL_SIZE "${WHL_SIZE}"

  # Build the wheel (with cpu flag)
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} ${PIP_WHL_FLAGS} --cpu

  for WHL_PATH in $(ls "${PIP_WHL_DIR}"/*.whl); do
    # change 10_15 to 10_14
    NEW_WHL_PATH=${WHL_PATH/macosx_10_15/macosx_10_14}
    mv "${WHL_PATH}" "${NEW_WHL_PATH}"
    WHL_PATH=${NEW_WHL_PATH}
  done
  # Deactivate Virtual Env
  deactivate || source deactivate
  rm -rf ${VENV_DIR}
  # Reset Python bin path
  export PYTHON_BIN_PATH="$(which python)"
}

function bazel_test_wheel {
  if [[ -z "${1}" ]]; then
    die "Missing wheel file path to install and test build"
  fi
  WHL_PATH=$1

# Create new Virtual Env for Testing
  VENV_DIR=".tf-venv"
  rm -rf "${VENV_DIR}"

  python -m venv ${VENV_DIR} && source ${VENV_DIR}/bin/activate
  export PYTHON_BIN_PATH="$(which python)"

  # Create Temp Dir to run the test
  TMP_DIR=$(mktemp -d)
  pushd "${TMP_DIR}"

  pip install "${WHL_PATH}"

  # Run a quick check on tensorflow installation.
  RET_VAL=$(python -c "import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)")

  # Check result to see if tensorflow is properly installed.
  if ! [[ ${RET_VAL} == *'(4,)'* ]]; then
    die "PIP test on virtualenv (non-clean) FAILED"
  fi

  # Return to original directory.
  popd
  rm -rf "${TMP_DIR}"

  # Run Bazel Test
  PY_MAJ_MINOR_VER=$(python -c "print(__import__('sys').version)" 2>&1 | awk '{ print $1 }' | head -n 1 | awk -F'.'  '{printf "%s%s\n", $1, $2 }')
  TF_TEST_FLAGS="--define=no_tensorflow_py_deps=true --test_lang_filters=py --test_output=errors --verbose_failures=true --keep_going --test_env=TF2_BEHAVIOR=1"
  TF_TEST_FILTER_TAGS="-nopip,-no_pip,-nomac,-no_mac,-mac_excluded,-no_oss,-oss_excluded,-oss_serial,-no_oss_py${PY_MAJ_MINOR_VER},-v1only,-gpu,-tpu,-benchmark-test"
  BAZEL_PARALLEL_TEST_FLAGS="--local_test_jobs=$(sysctl -n hw.ncpu)"

  # Install additional test requirements
  # TODO - Add these to setup.py test requirements
  pip install portpicker~=1.5.2 scipy~=1.7.2

  PIP_TEST_PREFIX=bazel_pip
  TEST_ROOT=$(pwd)/${PIP_TEST_PREFIX}
  rm -rf "$TEST_ROOT"
  mkdir -p "$TEST_ROOT"
  ln -s "$(pwd)"/tensorflow "$TEST_ROOT"/tensorflow

  bazel clean
  yes "" | ./configure
  # Adding quotes around TF_TEST_FLAGS variable leads to additional quotes in variable replacement
  # shellcheck disable=SC2086
  bazel test --build_tests_only \
    ${TF_TEST_FLAGS} \
    "${BAZEL_PARALLEL_TEST_FLAGS}" \
    --test_tag_filters="${TF_TEST_FILTER_TAGS}" \
    -k -- //bazel_pip/tensorflow/python/...

  unlink "${TEST_ROOT}"/tensorflow

  # Deactivate Virtual Env
  deactivate || source deactivate
  rm -rf ${VENV_DIR}
  # Reset Python bin path
  export PYTHON_BIN_PATH="$(which python)"
}

function upload_nightly_wheel {
  if [[ -z "${1}" ]]; then
    die "Missing wheel file path"
  fi
  WHL_PATH=$1
  # test the whl pip package
  chmod +x tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh
  ./tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh ${WHL_PATH}
  RETVAL=$?

  # Upload the PIP package if whl test passes.
  if [ ${RETVAL} -eq 0 ]; then
    echo "Basic PIP test PASSED, Uploading package: ${WHL_PATH}"
    python -m pip install 'twine ~= 3.2.0'
    python -m twine upload -r pypi-warehouse "${WHL_PATH}"
  else
    die "Basic PIP test FAILED, will not upload ${WHL_PATH} package"
  fi
}
