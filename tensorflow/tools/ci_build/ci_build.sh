#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# Usage: ci_build.sh <CONTAINER_TYPE> [--dockerfile <DOCKERFILE_PATH>]
#                    <COMMAND>
#
# CONTAINER_TYPE: Type of the docker container used the run the build:
#                 e.g., (cpu | gpu | gpu_clang | android | tensorboard)
#
# DOCKERFILE_PATH: (Optional) Path to the Dockerfile used for docker build.
#                  If this optional value is not supplied (via the
#                  --dockerfile flag), default Dockerfiles in the same
#                  directory as this script will be used.
#
# COMMAND: Command to be executed in the docker container, e.g.,
#          tensorflow/tools/ci_build/builds/pip.sh gpu -c opt --config=cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds/builds_common.sh"

# Get the command line arguments.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Dockerfile to be used in docker build
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}"
DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"

if [[ "$1" == "--dockerfile" ]]; then
  DOCKERFILE_PATH="$2"
  DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
  echo "Using custom Dockerfile path: ${DOCKERFILE_PATH}"
  echo "Using custom docker build context path: ${DOCKER_CONTEXT_PATH}"
  shift 2
fi

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
  die "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
fi

COMMAND=("$@")

# Validate command line arguments.
if [ "$#" -lt 1 ] || [ ! -e "${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}" ]; then
  supported_container_types=$( ls -1 ${SCRIPT_DIR}/Dockerfile.* | \
      sed -n 's/.*Dockerfile\.\([^\/]*\)/\1/p' | tr '\n' ' ' )
  >&2 echo "Usage: $(basename $0) CONTAINER_TYPE COMMAND"
  >&2 echo "       CONTAINER_TYPE can be one of [ ${supported_container_types}]"
  >&2 echo "       COMMAND is a command (with arguments) to run inside"
  >&2 echo "               the container."
  >&2 echo ""
  >&2 echo "Example (run all tests on CPU):"
  >&2 echo "$0 CPU bazel test //tensorflow/..."
  exit 1
fi

# Optional arguments - environment variables. For example:
# CI_DOCKER_EXTRA_PARAMS='-it --rm' CI_COMMAND_PREFIX='' tensorflow/tools/ci_build/ci_build.sh CPU /bin/bash
CI_TENSORFLOW_SUBMODULE_PATH="${CI_TENSORFLOW_SUBMODULE_PATH:-.}"
CI_COMMAND_PREFIX=("${CI_COMMAND_PREFIX[@]:-${CI_TENSORFLOW_SUBMODULE_PATH}/tensorflow/tools/ci_build/builds/with_the_same_user "\
"${CI_TENSORFLOW_SUBMODULE_PATH}/tensorflow/tools/ci_build/builds/configured ${CONTAINER_TYPE}}")

if [[ ! -z "${TF_BUILD_DISABLE_GCP}" ]] &&
   [[ "${TF_BUILD_DISABLE_GCP}" != "0" ]]; then
  CI_COMMAND_PREFIX+=("--disable-gcp")
fi

# cmake (CPU) builds do not require configuration.
if [[ "${CONTAINER_TYPE}" == "cmake" ]]; then
  CI_COMMAND_PREFIX=""
fi

# Use nvidia-docker if the container is GPU.
if [[ "${CONTAINER_TYPE}" == "gpu" ]] || [[ "${CONTAINER_TYPE}" == "gpu_clang" ]]; then
  DOCKER_BINARY="nvidia-docker"
else
  DOCKER_BINARY="docker"
fi

# Helper function to traverse directories up until given file is found.
function upsearch () {
  test / == "$PWD" && return || \
      test -e "$1" && echo "$PWD" && return || \
      cd .. && upsearch "$1"
}

# Set up WORKSPACE and BUILD_TAG. Jenkins will set them for you or we pick
# reasonable defaults if you run it outside of Jenkins.
WORKSPACE="${WORKSPACE:-$(upsearch WORKSPACE)}"
BUILD_TAG="${BUILD_TAG:-tf_ci}"

# Add extra params for cuda devices and libraries for GPU container.
# And clear them if we are not building for GPU.
if [[ "${CONTAINER_TYPE}" != "gpu" ]] && [[ "${CONTAINER_TYPE}" != "gpu_clang" ]]; then
  GPU_EXTRA_PARAMS=""
fi

# Determine the docker image name
DOCKER_IMG_NAME="${BUILD_TAG}.${CONTAINER_TYPE}"

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "CI_DOCKER_EXTRA_PARAMS: ${CI_DOCKER_EXTRA_PARAMS[@]}"
echo "COMMAND: ${COMMAND[@]}"
echo "CI_COMMAND_PREFIX: ${CI_COMMAND_PREFIX[@]}"
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "  (docker container name will be ${DOCKER_IMG_NAME})"
echo ""


# Build the docker container.
echo "Building container (${DOCKER_IMG_NAME})..."
docker build -t ${DOCKER_IMG_NAME} \
    -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

# Check docker build status
if [[ $? != "0" ]]; then
  die "ERROR: docker build failed. Dockerfile is at ${DOCKERFILE_PATH}"
fi

# Run the command inside the container.
echo "Running '${COMMAND[@]}' inside ${DOCKER_IMG_NAME}..."
mkdir -p ${WORKSPACE}/bazel-ci_build-cache
# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).
${DOCKER_BINARY} run --rm --pid=host \
    -v ${WORKSPACE}/bazel-ci_build-cache:${WORKSPACE}/bazel-ci_build-cache \
    -e "CI_BUILD_HOME=${WORKSPACE}/bazel-ci_build-cache" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "CI_TENSORFLOW_SUBMODULE_PATH=${CI_TENSORFLOW_SUBMODULE_PATH}" \
    -v ${WORKSPACE}:/workspace \
    -w /workspace \
    ${GPU_EXTRA_PARAMS} \
    ${CI_DOCKER_EXTRA_PARAMS[@]} \
    "${DOCKER_IMG_NAME}" \
    ${CI_COMMAND_PREFIX[@]} \
    ${COMMAND[@]}
