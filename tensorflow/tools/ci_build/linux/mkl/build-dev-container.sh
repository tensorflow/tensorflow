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
# Build a whl and container with Intel(R) MKL support
# Usage: build-dev-container.sh

DEBUG=1
DOCKER_BINARY="docker"
TMP_DIR=$(pwd)

# Helper function to traverse directories up until given file is found.
function upsearch () {
  test / == "$PWD" && return || \
      test -e "$1" && echo "$PWD" && return || \
      cd .. && upsearch "$1"
}

function debug()
{
  if [[ ${DEBUG} == 1 ]] ; then
    echo $1
  fi
}

function die()
{
  echo $1
  exit 1
}

# Set up WORKSPACE.
WORKSPACE="${WORKSPACE:-$(upsearch WORKSPACE)}"

ROOT_CONTAINER=${ROOT_CONTAINER:-tensorflow/tensorflow}
TF_ROOT_CONTAINER_TAG=${ROOT_CONTAINER_TAG:-devel}

# TF_BUILD_VERSION can be either a tag, branch, commit ID or PR number.
# For a PR, set TF_BUILD_VERSION_IS_PR="yes"
TF_BUILD_VERSION=${TF_DOCKER_BUILD_DEVEL_BRANCH:-master}
TF_BUILD_VERSION_IS_PR=${TF_DOCKER_BUILD_DEVEL_BRANCH_IS_PR:-no}
TF_REPO=${TF_REPO:-https://github.com/tensorflow/tensorflow}
FINAL_IMAGE_NAME=${TF_DOCKER_BUILD_IMAGE_NAME:-intel-mkl/tensorflow}
TF_DOCKER_BUILD_VERSION=${TF_DOCKER_BUILD_VERSION:-nightly}
BUILD_AVX_CONTAINERS=${BUILD_AVX_CONTAINERS:-no}
BUILD_AVX2_CONTAINERS=${BUILD_AVX2_CONTAINERS:-no}
BUILD_SKX_CONTAINERS=${BUILD_SKX_CONTAINERS:-no}
BUILD_CLX_CONTAINERS=${BUILD_CLX_CONTAINERS:-no}
CONTAINER_PORT=${TF_DOCKER_BUILD_PORT:-8888}
BUILD_TF_V2_CONTAINERS=${BUILD_TF_V2_CONTAINERS:-yes}
BUILD_TF_BFLOAT16_CONTAINERS=${BUILD_TF_BFLOAT16_CONTAINERS:-no}
ENABLE_SECURE_BUILD=${ENABLE_SECURE_BUILD:-no}
BAZEL_VERSION=${BAZEL_VERSION}
BUILD_PY2_CONTAINERS=${BUILD_PY2_CONTAINERS:-no}
ENABLE_DNNL1=${ENABLE_DNNL1:-no}
ENABLE_HOROVOD=${ENABLE_HOROVOD:-no}
OPENMPI_VERSION=${OPENMPI_VERSION}
OPENMPI_DOWNLOAD_URL=${OPENMPI_DOWNLOAD_URL}
HOROVOD_VERSION=${HOROVOD_VERSION}
IS_NIGHTLY=${IS_NIGHTLY:-no}

debug "ROOT_CONTAINER=${ROOT_CONTAINER}"
debug "TF_ROOT_CONTAINER_TAG=${TF_ROOT_CONTAINER_TAG}"
debug "TF_BUILD_VERSION=${TF_BUILD_VERSION}"
debug "TF_BUILD_VERSION_IS_PR=${TF_BUILD_VERSION_IS_PR}"
debug "FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME}"
debug "TF_DOCKER_BUILD_VERSION=${TF_DOCKER_BUILD_VERSION}"
debug "BUILD_AVX_CONTAINERS=${BUILD_AVX_CONTAINERS}"
debug "BUILD_AVX2_CONTAINERS=${BUILD_AVX2_CONTAINERS}"
debug "BUILD_SKX_CONTAINERS=${BUILD_SKX_CONTAINERS}"
debug "BUILD_CLX_CONTAINERS=${BUILD_CLX_CONTAINERS}"
debug "BUILD_TF_V2_CONTAINERS=${BUILD_TF_V2_CONTAINERS}"
debug "BUILD_TF_BFLOAT16_CONTAINERS=${BUILD_TF_BFLOAT16_CONTAINERS}"
debug "ENABLE_SECURE_BUILD=${ENABLE_SECURE_BUILD}"
debug "TMP_DIR=${TMP_DIR}"
debug "BAZEL_VERSION=${BAZEL_VERSION}"
debug "BUILD_PY2_CONTAINERS=${BUILD_PY2_CONTAINERS}"
debug "ENABLE_DNNL1=${ENABLE_DNNL1}"
debug "ENABLE_HOROVOD=${ENABLE_HOROVOD}"
debug "OPENMPI_VERSION=${OPENMPI_VERSION}"
debug "OPENMPI_DOWNLOAD_URL=${OPENMPI_DOWNLOAD_URL}"
debug "HOROVOD_VERSION=${HOROVOD_VERSION}"
debug "IS_NIGHTLY=${IS_NIGHTLY}"

function build_container()
{
  if [[ $# -lt 2 ]]; then
    die "Usage: build_container <TEMP_IMAGE_NAME> <TF_DOCKER_BUILD_ARGS>."
  fi
  TEMP_IMAGE_NAME=${1}
  debug "TEMP_IMAGE_NAME=${TEMP_IMAGE_NAME}"
  shift
  TF_DOCKER_BUILD_ARGS=("${@}")

  # Add the proxy info build args. This will be later on passed to docker as
  # --build-arg so that users behind corporate proxy can build the images
  TF_DOCKER_BUILD_ARGS+=("--build-arg http_proxy=${http_proxy}")
  TF_DOCKER_BUILD_ARGS+=("--build-arg https_proxy=${https_proxy}")
  TF_DOCKER_BUILD_ARGS+=("--build-arg socks_proxy=${socks_proxy}")
  TF_DOCKER_BUILD_ARGS+=("--build-arg no_proxy=${no_proxy}")
  # In general having uppercase proxies is a good idea because different
  # applications running inside Docker may only honor uppercase proxies
  TF_DOCKER_BUILD_ARGS+=("--build-arg HTTP_PROXY=${HTTP_PROXY}")
  TF_DOCKER_BUILD_ARGS+=("--build-arg HTTPS_PROXY=${HTTPS_PROXY}")
  TF_DOCKER_BUILD_ARGS+=("--build-arg SOCKS_PROXY=${SOCKS_PROXY}")
  TF_DOCKER_BUILD_ARGS+=("--build-arg NO_PROXY=${NO_PROXY}")

  #Add --config=v2 build arg for TF v2
  if [[ ${BUILD_TF_V2_CONTAINERS} == "no" ]]; then
    TF_DOCKER_BUILD_ARGS+=("--build-arg CONFIG_V2_DISABLE=--disable-v2")
  fi

  #Add build arg for bfloat16 build
  if [[ ${BUILD_TF_BFLOAT16_CONTAINERS} == "yes" ]]; then
    TF_DOCKER_BUILD_ARGS+=("--build-arg CONFIG_BFLOAT16_BUILD=--enable-bfloat16")
  fi

  #Add build arg for Secure Build
  if [[ ${ENABLE_SECURE_BUILD} == "yes" ]]; then
    TF_DOCKER_BUILD_ARGS+=("--build-arg ENABLE_SECURE_BUILD=--secure-build")
  fi

  # Add build arg for DNNL1
  if [[ ${ENABLE_DNNL1} == "yes" ]]; then
    TF_DOCKER_BUILD_ARGS+=("--build-arg ENABLE_DNNL1=--enable-dnnl1")
  fi

  # BAZEL Version
  if [[ ${BAZEL_VERSION} != "" ]]; then
    TF_DOCKER_BUILD_ARGS+=("--build-arg BAZEL_VERSION=${BAZEL_VERSION}")
  fi

  # Add build arg for installing OpenMPI/Horovod
  if [[ ${ENABLE_HOROVOD} == "yes" ]]; then
    TF_DOCKER_BUILD_ARGS+=("--build-arg ENABLE_HOROVOD=${ENABLE_HOROVOD}")
    TF_DOCKER_BUILD_ARGS+=("--build-arg OPENMPI_VERSION=${OPENMPI_VERSION}")
    TF_DOCKER_BUILD_ARGS+=("--build-arg OPENMPI_DOWNLOAD_URL=${OPENMPI_DOWNLOAD_URL}")
    TF_DOCKER_BUILD_ARGS+=("--build-arg HOROVOD_VERSION=${HOROVOD_VERSION}")
  fi

  # Add build arg --nightly_flag for the nightly build
  if [[ ${IS_NIGHTLY} == "yes" ]]; then
    TF_DOCKER_BUILD_ARGS+=("--build-arg TF_NIGHTLY_FLAG=--nightly_flag")
  fi

  # Perform docker build
  debug "Building docker image with image name and tag: ${TEMP_IMAGE_NAME}"
  CMD="${DOCKER_BINARY} build ${TF_DOCKER_BUILD_ARGS[@]} --no-cache --pull -t ${TEMP_IMAGE_NAME} -f Dockerfile.devel-mkl ."
  debug "CMD=${CMD}"
  ${CMD}

  if [[ $? == "0" ]]; then
    debug "${DOCKER_BINARY} build of ${TEMP_IMAGE_NAME} succeeded"
  else
    die "FAIL: ${DOCKER_BINARY} build of ${TEMP_IMAGE_NAME} failed"
  fi
}

function test_container()
{
  if [[ "$#" != "1" ]]; then
    die "Usage: ${FUNCNAME} <TEMP_IMAGE_NAME>"
  fi

  TEMP_IMAGE_NAME=${1}

  # Make sure that there is no other containers of the same image running
  if "${DOCKER_BINARY}" ps | grep -q "${TEMP_IMAGE_NAME}"; then
    die "ERROR: It appears that there are docker containers of the image "\
  "${TEMP_IMAGE_NAME} running. Please stop them before proceeding"
  fi

  # Start a docker container from the newly-built docker image
  DOCKER_RUN_LOG="${TMP_DIR}/docker_run.log"
  debug "  Log file is at: ${DOCKER_RUN_LOG}"

  debug "Running docker container from image ${TEMP_IMAGE_NAME}..."
  RUN_CMD="${DOCKER_BINARY} run --rm -d -p ${CONTAINER_PORT}:${CONTAINER_PORT} ${TEMP_IMAGE_NAME} tail -f /dev/null 2>&1 > ${DOCKER_RUN_LOG}"
  debug "RUN_CMD=${RUN_CMD}"
  ${RUN_CMD}

  # Get the container ID
  CONTAINER_ID=""
  while [[ -z ${CONTAINER_ID} ]]; do
    sleep 1
    debug "Polling for container ID..."
    CONTAINER_ID=$("${DOCKER_BINARY}" ps | grep "${TEMP_IMAGE_NAME}" | awk '{print $1}')
  done

  debug "ID of the running docker container: ${CONTAINER_ID}"

  debug "Performing basic sanity checks on the running container..."
  {
    ${DOCKER_BINARY} exec ${CONTAINER_ID} bash -c "${PYTHON} -c 'from tensorflow.python import _pywrap_util_port; print(_pywrap_util_port.IsMklEnabled())'"
    echo "PASS: MKL enabled test in ${TEMP_IMAGE_NAME}"
  } || {
    ${DOCKER_BINARY} exec ${CONTAINER_ID} bash -c "${PYTHON} -c 'from tensorflow.python import pywrap_tensorflow; print(pywrap_tensorflow.IsMklEnabled())'"
    echo "PASS: Old MKL enabled in ${TEMP_IMAGE_NAME}"
  } || {
    die "FAIL: MKL enabled test in ${TEMP_IMAGE_NAME}"
  }

  # Test to check if horovod is installed successfully
  if [[ ${ENABLE_HOROVOD} == "yes" ]]; then
      debug "Test horovod in the container..."
      ${DOCKER_BINARY} exec ${CONTAINER_ID} bash -c "${PYTHON} -c 'import horovod.tensorflow as hvd;'"
      if [[ $? == "0" ]]; then
          echo "PASS: HOROVOD installation test in ${TEMP_IMAGE_NAME}"
      else
          die "FAIL: HOROVOD installation test in ${TEMP_IMAGE_NAME}"
      fi
  fi
  
  # Stop the running docker container
  sleep 1
  "${DOCKER_BINARY}" stop --time=0 ${CONTAINER_ID}
}

function checkout_tensorflow()
{
  if [[ "$#" != "3" ]]; then
    die "Usage: ${FUNCNAME} <REPO_URL> <BRANCH/TAG/COMMIT-ID/PR-ID> <TF_BUILD_VERSION_IS_PR>"
  fi

  TF_REPO="${1}"
  TF_BUILD_VERSION="${2}"
  TF_BUILD_VERSION_IS_PR="${3}"
  TENSORFLOW_DIR="tensorflow"

  debug "Checking out ${TF_REPO}:${TF_BUILD_VERSION} into ${TENSORFLOW_DIR}"

  # Clean any existing tensorflow sources
  rm -rf "${TENSORFLOW_DIR}"

  git clone ${TF_REPO} ${TENSORFLOW_DIR}
  cd ${TENSORFLOW_DIR}
  if [[ "${TF_BUILD_VERSION_IS_PR}" == "yes" ]]; then
    # If TF_BUILD_VERSION is a PR number, then fetch first
    git fetch origin pull/${TF_BUILD_VERSION}/head:pr-${TF_BUILD_VERSION}
    git checkout pr-${TF_BUILD_VERSION}
  else
    git checkout ${TF_BUILD_VERSION}
  fi
  if [ $? -ne 0 ]; then
    die "Unable to find ${TF_BUILD_VERSION} on ${TF_REPO}"
  fi
  cd ..
}

function tag_container()
{
  # Apply the final image name and tag
  TEMP_IMAGE_NAME="${1}"
  FINAL_IMG="${2}"

  DOCKER_VER=$("${DOCKER_BINARY}" version | grep Version | head -1 | awk '{print $NF}')
  if [[ -z "${DOCKER_VER}" ]]; then
    die "ERROR: Failed to determine ${DOCKER_BINARY} version"
  fi
  DOCKER_MAJOR_VER=$(echo "${DOCKER_VER}" | cut -d. -f 1)
  DOCKER_MINOR_VER=$(echo "${DOCKER_VER}" | cut -d. -f 2)

  FORCE_TAG=""
  if [[ "${DOCKER_MAJOR_VER}" -le 1 ]] && \
    [[ "${DOCKER_MINOR_VER}" -le 9 ]]; then
    FORCE_TAG="--force"
  fi

  "${DOCKER_BINARY}" tag ${FORCE_TAG} "${TEMP_IMAGE_NAME}" "${FINAL_IMG}" || \
      die "Failed to tag intermediate docker image ${TEMP_IMAGE_NAME} as ${FINAL_IMG}"

  debug "Successfully tagged docker image: ${FINAL_IMG}"
}

PYTHON_VERSIONS=("python3")
if [[ ${BUILD_PY2_CONTAINERS} == "yes" ]]; then
  PYTHON_VERSIONS+=("python")
fi

PLATFORMS=()
if [[ ${BUILD_AVX_CONTAINERS} == "yes" ]]; then
  PLATFORMS+=("sandybridge")
fi

if [[ ${BUILD_AVX2_CONTAINERS} == "yes" ]]; then
  PLATFORMS+=("haswell")
fi

if [[ ${BUILD_SKX_CONTAINERS} == "yes" ]]; then
  PLATFORMS+=("skylake")
fi

if [[ ${BUILD_CLX_CONTAINERS} == "yes" ]]; then
  PLATFORMS+=("icelake")
fi

# Checking out sources needs to be done only once
checkout_tensorflow "${TF_REPO}" "${TF_BUILD_VERSION}" "${TF_BUILD_VERSION_IS_PR}"

for PLATFORM in "${PLATFORMS[@]}"
do
  for PYTHON in "${PYTHON_VERSIONS[@]}"
  do
    # Clear the build args array
    TF_DOCKER_BUILD_ARGS=("--build-arg TARGET_PLATFORM=${PLATFORM}")
    TF_DOCKER_BUILD_ARGS+=("--build-arg ROOT_CONTAINER=${ROOT_CONTAINER}")
    FINAL_TAG="${TF_DOCKER_BUILD_VERSION}"
    ROOT_CONTAINER_TAG="${TF_ROOT_CONTAINER_TAG}"

      if [[ ${PLATFORM} == "haswell" ]]; then
        FINAL_TAG="${FINAL_TAG}-avx2"
      fi

      if [[ ${PLATFORM} == "skylake" ]]; then
        FINAL_TAG="${FINAL_TAG}-avx512"
      fi

      if [[ ${PLATFORM} == "icelake" ]]; then
        FINAL_TAG="${FINAL_TAG}-avx512-VNNI"
      fi

      # Add -devel-mkl to the image tag
      FINAL_TAG="${FINAL_TAG}-devel-mkl"
      if [[ "${PYTHON}" == "python3" ]]; then
        TF_DOCKER_BUILD_ARGS+=("--build-arg WHL_DIR=/tmp/pip3")
        TF_DOCKER_BUILD_ARGS+=("--build-arg PIP=pip3")
      fi

      TF_DOCKER_BUILD_ARGS+=("--build-arg PYTHON=${PYTHON}")
      TF_DOCKER_BUILD_ARGS+=("--build-arg ROOT_CONTAINER_TAG=${ROOT_CONTAINER_TAG}")

      # Intermediate image name with tag
      TEMP_IMAGE_NAME="${USER}/tensorflow:${FINAL_TAG}"
      build_container "${TEMP_IMAGE_NAME}" "${TF_DOCKER_BUILD_ARGS[@]}"
      test_container "${TEMP_IMAGE_NAME}"
      tag_container "${TEMP_IMAGE_NAME}" "${FINAL_IMAGE_NAME}:${FINAL_TAG}"
  done
done
