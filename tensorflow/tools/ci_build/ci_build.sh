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

# Validate command line arguments.
if [ "$#" -lt 1 ] || [[ ! "${CONTAINER_TYPE}" =~ ^(cpu|gpu|android)$ ]]; then
  >&2 echo "Usage: $(basename $0) CONTAINER_TYPE COMMAND"
  >&2 echo "       CONTAINER_TYPE can be 'CPU' or 'GPU'"
  >&2 echo "       COMMAND is a command (with arguments) to run inside"
  >&2 echo "               the container."
  >&2 echo ""
  >&2 echo "Example (run all tests on CPU):"
  >&2 echo "$0 CPU bazel test //tensorflow/..."
  exit 1
fi


# Figure out the directory where this script is.
SCRIPT_DIR=$( cd ${0%/*} && pwd -P )

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

# Additional configuration. You can customize it by modifying
# env variable.
EXTRA_DEPS_DIR="${EXTRA_DEPS_DIR:-${HOME}/.tensorflow_extra_deps}"


# Print arguments.
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "COMMAND: ${COMMAND[@]}"
echo "WORKSAPCE: ${WORKSPACE}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "  (docker container name will be ${BUILD_TAG}.${CONTAINER_TYPE})"
echo "EXTRA_DEPS_DIR: ${EXTRA_DEPS_DIR}"
echo ""

# Build the docker containers.
echo "Building CPU container (${BUILD_TAG}.cpu)..."
docker build -t ${BUILD_TAG}.cpu -f ${SCRIPT_DIR}/Dockerfile.cpu ${SCRIPT_DIR}
if [ "${CONTAINER_TYPE}" != "cpu" ]; then
  echo "Building container ${BUILD_TAG}.${CONTAINER_TYPE}..."
  tmp_dockerfile="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}.${BUILD_TAG}"
  # we need to generate temporary dockerfile with overwritten FROM directive
  sed "s/^FROM .*/FROM ${BUILD_TAG}.cpu/" \
      ${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE} > ${tmp_dockerfile}
  docker build -t ${BUILD_TAG}.${CONTAINER_TYPE} \
      -f ${tmp_dockerfile} ${SCRIPT_DIR}
  rm ${tmp_dockerfile}
fi


# Run the command inside the container.
echo "Running '${COMMAND[@]}' inside ${BUILD_TAG}.${CONTAINER_TYPE}..."
mkdir -p ${WORKSPACE}/bazel-user-cache-for-docker
docker run \
    -v ${WORKSPACE}/bazel-user-cache-for-docker:/root/.cache \
    -v ${WORKSPACE}:/tensorflow \
    -v ${EXTRA_DEPS_DIR}:/tensorflow_extra_deps \
    -w /tensorflow \
    ${BUILD_TAG}.${CONTAINER_TYPE} \
    "${COMMAND[@]}"
