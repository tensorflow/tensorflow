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


# Print arguments.
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "COMMAND: ${COMMAND[@]}"
echo "WORKSAPCE: ${WORKSPACE}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "  (docker container name will be ${BUILD_TAG}.${CONTAINER_TYPE})"
echo ""


# Build the docker container.
echo "Building container (${BUILD_TAG}.${CONTAINER_TYPE})..."
docker build -t ${BUILD_TAG}.${CONTAINER_TYPE} \
    -f ${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE} ${SCRIPT_DIR}


# Run the command inside the container.
echo "Running '${COMMAND[@]}' inside ${BUILD_TAG}.${CONTAINER_TYPE}..."
mkdir -p ${WORKSPACE}/bazel-ci_build-cache
docker run \
    -v ${WORKSPACE}/bazel-ci_build-cache:${WORKSPACE}/bazel-ci_build-cache \
    -e "CI_BUILD_HOME=${WORKSPACE}/bazel-ci_build-cache" \
    -e "CI_BUILD_USER=${USER}" \
    -e "CI_BUILD_UID=$(id -u $USER)" \
    -e "CI_BUILD_GROUP=$(id -g --name $USER)" \
    -e "CI_BUILD_GID=$(id -g $USER)" \
    -v ${WORKSPACE}:/tensorflow \
    -w /tensorflow \
    ${BUILD_TAG}.${CONTAINER_TYPE} \
    "tensorflow/tools/ci_build/builds/with_the_same_user" \
        "tensorflow/tools/ci_build/builds/configured" \
        "${CONTAINER_TYPE}" "${COMMAND[@]}"
