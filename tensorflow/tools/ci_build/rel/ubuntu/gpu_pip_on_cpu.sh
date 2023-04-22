#!/bin/bash
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
set -e
set -x

source tensorflow/tools/ci_build/release/common.sh

install_ubuntu_16_python_pip_deps python3.6
# Update Bazel to the desired version
install_bazelisk

export LD_LIBRARY_PATH="/usr/local/cuda:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/tensorrt/lib"

########################
## Build GPU pip package
########################
export PYTHON_BIN_PATH=$(which python3.6)
set +e  # piping `yes` can raise non-breaking errors; ignore them.
yes "" | "$PYTHON_BIN_PATH" configure.py
set -e

bazel build \
  --config=release_gpu_linux \
  --repo_env=PYTHON_BIN_PATH="$(which python3.6)" \
  tensorflow/tools/pip_package:build_pip_package

# Set TF nightly flag so we get the proper version of estimator
if [[ "$IS_NIGHTLY" == 1 ]]; then
  NIGHTLY_FLAG="--nightly_flag"
fi

PIP_WHL_DIR=whl
mkdir -p ${PIP_WHL_DIR}
PIP_WHL_DIR=$(readlink -f ${PIP_WHL_DIR})  # Get absolute path
bazel-bin/tensorflow/tools/pip_package/build_pip_package "${PIP_WHL_DIR}" "${NIGHTLY_FLAG}"
WHL_PATH=$(ls "${PIP_WHL_DIR}"/*.whl)

cp "${WHL_PATH}" "$(pwd)"/.
chmod +x tensorflow/tools/ci_build/builds/docker_cpu_pip.sh
docker run -e "BAZEL_VERSION=${BAZEL_VERSION}" -e "CI_BUILD_USER=$(id -u -n)" -e "CI_BUILD_UID=$(id -u)"  -e "CI_BUILD_GROUP=$(id -g -n)" -e "CI_BUILD_GID=$(id -g)"  -e "CI_BUILD_HOME=/bazel_pip" -v "$(pwd)":/bazel_pip tensorflow/tensorflow:devel "./bazel_pip/tensorflow/tools/ci_build/builds/with_the_same_user" "./bazel_pip/tensorflow/tools/ci_build/builds/docker_cpu_pip.sh"
