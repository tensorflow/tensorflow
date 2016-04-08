#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
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


# Get the command line arguments.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1
COMMAND=("$@")

# Figure out the directory where this script is.
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )

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
if [[ "${CI_DOCKER_EXTRA_PARAMS}" != *"--rm"* ]]; then
  CI_DOCKER_EXTRA_PARAMS="--rm ${CI_DOCKER_EXTRA_PARAMS}"
fi
CI_TENSORFLOW_SUBMODULE_PATH="${CI_TENSORFLOW_SUBMODULE_PATH:-.}"
CI_COMMAND_PREFIX=("${CI_COMMAND_PREFIX[@]:-${CI_TENSORFLOW_SUBMODULE_PATH}/tensorflow/tools/ci_build/builds/with_the_same_user ${CI_TENSORFLOW_SUBMODULE_PATH}/tensorflow/tools/ci_build/builds/configured ${CONTAINER_TYPE}}")


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
if [ "${CONTAINER_TYPE}" == "gpu" ]; then
  devices=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
  libs=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
  GPU_EXTRA_PARAMS="${devices} ${libs}"

  # GPU pip tests-on-install should avoid using concurrent jobs due to GPU
  # resource contention
  GPU_EXTRA_PARAMS="${GPU_EXTRA_PARAMS} -e TF_BUILD_SERIAL_INSTALL_TESTS=1"
else
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
echo "WORKSAPCE: ${WORKSPACE}"
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
    -f ${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE} ${SCRIPT_DIR}

# Run the command inside the container.
echo "Running '${COMMAND[@]}' inside ${DOCKER_IMG_NAME}..."
mkdir -p ${WORKSPACE}/bazel-ci_build-cache
docker run \
    -v ${WORKSPACE}/bazel-ci_build-cache:${WORKSPACE}/bazel-ci_build-cache \
    -e "CI_BUILD_HOME=${WORKSPACE}/bazel-ci_build-cache" \
    -e "CI_BUILD_USER=$(id -u --name)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g --name)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "CI_TENSORFLOW_SUBMODULE_PATH=${CI_TENSORFLOW_SUBMODULE_PATH}" \
    -v ${WORKSPACE}:/workspace \
    -w /workspace \
    ${GPU_EXTRA_PARAMS} \
    ${CI_DOCKER_EXTRA_PARAMS[@]} \
    "${DOCKER_IMG_NAME}" \
    ${CI_COMMAND_PREFIX[@]} \
    ${COMMAND[@]}
