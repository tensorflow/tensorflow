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
# External `common.sh`

# Keep in sync with tensorflow_estimator and configure.py.
# LINT.IfChange
LATEST_BAZEL_VERSION=0.26.1
# LINT.ThenChange(
#   //tensorflow/opensource_only/configure.py,
#   //tensorflow_estimator/google/kokoro/common.sh,
#   //tensorflow/tools/ci_build/install/install_bazel.sh,
#   //tensorflow/tools/ci_build/install/install_bazel_from_source.sh)

# LINT.IfChange
function die() {
  echo "$@" 1>&2 ; exit 1;
}

# A small utility to run the command and only print logs if the command fails.
# On success, all logs are hidden.
function readable_run {
  # Disable debug mode to avoid printing of variables here.
  set +x
  result=$("$@" 2>&1) || die "$result"
  echo "$@"
  echo "Command completed successfully at $(date)"
  set -x
}
# LINT.ThenChange(
# ) # common_.sh

# LINT.IfChange
# Redirect bazel output dir b/73748835
function set_bazel_outdir {
  mkdir -p /tmpfs/bazel_output
  export TEST_TMPDIR=/tmpfs/bazel_output
}

# Run flaky functions with retries.
# run_with_retry cmd
function run_with_retry {
  eval "$1"
  # If the command fails retry again in 60 seconds.
  if [[ $? -ne 0 ]]; then
    sleep 60
    eval "$1"
  fi
}

# Install the given bazel version on linux
function update_bazel_linux {
  if [[ -z "$1" ]]; then
    BAZEL_VERSION=${LATEST_BAZEL_VERSION}
  else
    BAZEL_VERSION=$1
  fi
  rm -rf ~/bazel
  mkdir ~/bazel

  pushd ~/bazel
  readable_run wget https://github.com/bazelbuild/bazel/releases/download/"${BAZEL_VERSION}"/bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
  chmod +x bazel-*.sh
  ./bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh --user
  rm bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
  popd

  PATH="/home/kbuilder/bin:$PATH"
  set_bazel_outdir
  which bazel
  bazel version
}
# LINT.ThenChange(
#   //tensorflow_estimator/google/kokoro/common.sh)

# LINT.IfChange
# Install the given bazel version on macos
function update_bazel_macos {
  if [[ -z "$1" ]]; then
    BAZEL_VERSION=${LATEST_BAZEL_VERSION}
  else
    BAZEL_VERSION=$1
  fi
  BAZEL_COMMAND="curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh -O && \
  chmod +x bazel-*.sh && ./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh --user && \
  rm -f bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh"
  # If the bazel update fails retry again in 60 seconds.
  run_with_retry "${BAZEL_COMMAND}"
  # Add new bazel installation to path
  PATH="/Users/kbuilder/bin:$PATH"
  set_bazel_outdir
  which bazel
  bazel version
}

function install_pip2 {
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  sudo python2 get-pip.py
}

function install_pip3.5 {
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  sudo python3.5 get-pip.py
}

function install_pip_deps {
  SUDO_CMD=""
  PIP_CMD="pip"

  while true; do
    if [[ -z "${1}" ]]; then
      break
    fi
    if [[ "$1" == "sudo" ]]; then
      SUDO_CMD="sudo "
    elif [[ "$1" == "pip"* ]]; then
      PIP_CMD="$1"
    fi
    shift
  done

  # TODO(aselle): Change all these to be --user instead of sudo.
  # TODO(hyey): Add back IfChange lint check (b/143530103).
  # ===================================================================
  # Please change dependencies in `install_ubuntu_16_pip_deps` as well.
  # ===================================================================
  ${SUDO_CMD} ${PIP_CMD} install keras_applications==1.0.8 --no-deps
  ${SUDO_CMD} ${PIP_CMD} install keras_preprocessing==1.1.0 --no-deps
  ${SUDO_CMD} ${PIP_CMD} install gast==0.2.2
  ${SUDO_CMD} ${PIP_CMD} install h5py==2.8.0
  ${SUDO_CMD} ${PIP_CMD} install six==1.12.0
  ${SUDO_CMD} ${PIP_CMD} install grpcio
  ${SUDO_CMD} ${PIP_CMD} install portpicker
  ${SUDO_CMD} ${PIP_CMD} install scipy
  ${SUDO_CMD} ${PIP_CMD} install scikit-learn==0.20.3
  ${SUDO_CMD} ${PIP_CMD} install --upgrade tb-nightly
  ${PIP_CMD} install --user --upgrade attrs
  ${PIP_CMD} install --user --upgrade tf-estimator-nightly
  ${PIP_CMD} install --user --upgrade "future>=0.17.1"
  # ===================================================================
}

function install_ubuntu_16_pip_deps {
  PIP_CMD="pip"

  while true; do
    if [[ -z "${1}" ]]; then
      break
    fi
    if [[ "$1" == "pip"* ]]; then
      PIP_CMD="$1"
    fi
    shift
  done

  # TODO(hyey): Add back IfChange lint check (b/143530103).
  # ===================================================================
  # Please change dependencies in `install_pip_deps` as well.
  # ===================================================================
  "${PIP_CMD}" install --user --upgrade attrs
  "${PIP_CMD}" install keras_applications==1.0.8 --no-deps --user
  "${PIP_CMD}" install keras_preprocessing==1.1.0 --no-deps --user
  "${PIP_CMD}" install numpy==1.14.5 --user
  "${PIP_CMD}" install --user --upgrade "future>=0.17.1"
  "${PIP_CMD}" install gast==0.2.2 --user
  "${PIP_CMD}" install h5py==2.8.0 --user
  "${PIP_CMD}" install six==1.12.0 --user
  "${PIP_CMD}" install grpcio --user
  "${PIP_CMD}" install portpicker --user
  "${PIP_CMD}" install scipy --user
  "${PIP_CMD}" install scikit-learn --user
  "${PIP_CMD}" install --user --upgrade tf-estimator-nightly
  "${PIP_CMD}" install --user --upgrade tb-nightly
  # ===================================================================
}

function install_macos_pip_deps {
  SUDO_CMD=""
  PIP_CMD="pip"

  while true; do
    if [[ -z "${1}" ]]; then
      break
    fi
    if [[ "$1" == "sudo" ]]; then
      SUDO_CMD="sudo "
    elif [[ "$1" == "pip3.7" ]]; then
      PIP_CMD="python3.7 -m pip"
      SUDO_CMD="sudo -H "
    elif [[ "$1" == "pip"* ]]; then
      PIP_CMD="$1"
    fi
    shift
  done

   # High Sierra pip for Python2.7 installs don't work as expected.
   if [[ "${PIP_CMD}" == "pip" ]]; then
    PIP_CMD="python -m pip"
    SUDO_CMD="sudo -H "
   fi

  # TODO(aselle): Change all these to be --user instead of sudo.
  ${SUDO_CMD} ${PIP_CMD} install --upgrade setuptools==39.1.0
  ${SUDO_CMD} ${PIP_CMD} install keras_applications==1.0.8 --no-deps
  ${SUDO_CMD} ${PIP_CMD} install keras_preprocessing==1.1.0 --no-deps
  ${SUDO_CMD} ${PIP_CMD} install --upgrade mock portpicker scipy grpcio
  ${SUDO_CMD} ${PIP_CMD} install six==1.12.0
  ${SUDO_CMD} ${PIP_CMD} install scikit-learn==0.20.3
  ${SUDO_CMD} ${PIP_CMD} install numpy==1.14.5
  ${SUDO_CMD} ${PIP_CMD} install gast==0.2.2
  ${SUDO_CMD} ${PIP_CMD} install h5py==2.8.0
  ${SUDO_CMD} ${PIP_CMD} install --upgrade grpcio
  ${SUDO_CMD} ${PIP_CMD} install --upgrade tensorboard==1.15
  ${PIP_CMD} install --user --upgrade attrs
  ${PIP_CMD} install --user --upgrade tensorflow_estimator==1.15
  ${PIP_CMD} install --user --upgrade "future>=0.17.1"
}

function maybe_skip_v1 {
  # If we are building with v2 by default, skip tests with v1only tag.
  if grep -q "build --config=v2" ".bazelrc"; then
    echo ",-v1only"
  else
    echo ""
  fi
}

# Copy and rename a wheel to a new project name.
# Usage: copy_to_new_project_name <whl_path> <new_project_name>, for example
# copy_to_new_project_name test_dir/tf_nightly-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl tf_nightly_cpu
# will create a wheel with the same tags, but new project name under the same
# directory at
# test_dir/tf_nightly_cpu-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl
function copy_to_new_project_name {
  WHL_PATH="$1"
  NEW_PROJECT_NAME="$2"

  ORIGINAL_WHL_NAME=$(basename "${WHL_PATH}")
  ORIGINAL_WHL_DIR=$(realpath "$(dirname "${WHL_PATH}")")
  ORIGINAL_PROJECT_NAME="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 1)"
  FULL_TAG="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 2-)"
  NEW_WHL_NAME="${NEW_PROJECT_NAME}-${FULL_TAG}"
  VERSION="$(echo "${FULL_TAG}" | cut -d '-' -f 1)"

  TMP_DIR="$(mktemp -d)"
  cp "${WHL_PATH}" "${TMP_DIR}"
  pushd "${TMP_DIR}"
  unzip -q "${ORIGINAL_WHL_NAME}"

  ORIGINAL_WHL_DIR_PREFIX="${ORIGINAL_PROJECT_NAME}-${VERSION}"
  NEW_WHL_DIR_PREFIX="${NEW_PROJECT_NAME}-${VERSION}"
  mv "${ORIGINAL_WHL_DIR_PREFIX}.dist-info" "${NEW_WHL_DIR_PREFIX}.dist-info"
  mv "${ORIGINAL_WHL_DIR_PREFIX}.data" "${NEW_WHL_DIR_PREFIX}.data"
  sed -i.bak "s/${ORIGINAL_PROJECT_NAME}/${NEW_PROJECT_NAME}/g" "${NEW_WHL_DIR_PREFIX}.dist-info/RECORD"

  ORIGINAL_PROJECT_NAME_DASH="${ORIGINAL_PROJECT_NAME//_/-}"
  NEW_PROJECT_NAME_DASH="${NEW_PROJECT_NAME//_/-}"
  sed -i.bak "s/${ORIGINAL_PROJECT_NAME_DASH}/${NEW_PROJECT_NAME_DASH}/g" "${NEW_WHL_DIR_PREFIX}.dist-info/METADATA"

  zip -rq "${NEW_WHL_NAME}" "${NEW_WHL_DIR_PREFIX}.dist-info" "${NEW_WHL_DIR_PREFIX}.data" "tensorflow" "tensorflow_core"
  mv "${NEW_WHL_NAME}" "${ORIGINAL_WHL_DIR}"
  popd
  rm -rf "${TMP_DIR}"
}
# LINT.ThenChange(
# ) # common.sh
