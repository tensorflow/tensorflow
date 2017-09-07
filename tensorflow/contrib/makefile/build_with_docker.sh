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
# Usage: build_with_docker.sh
#
#   Will build a docker container, and run the makefile build inside that
#   container.
#

# Make sure we're in the correct directory, at the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${SCRIPT_DIR}/../../../"
cd ${WORKSPACE} || exit 1

DOCKER_IMG_NAME="tf-make-base"
DOCKER_CONTEXT_PATH="${WORKSPACE}tensorflow/contrib/makefile/"
DOCKERFILE_PATH="${DOCKER_CONTEXT_PATH}Dockerfile"

# Build the docker image.
echo "Building image ${DOCKER_IMG_NAME}..."
docker build -t ${DOCKER_IMG_NAME} \
    -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

# Check docker build command status.
if [[ $? != "0" ]]; then
  echo "ERROR: docker build failed. Dockerfile is at ${DOCKERFILE_PATH}"
  exit 1
fi

COMMAND="tensorflow/contrib/makefile/build_all_linux.sh"

# Run the command inside the container.
echo "Running ${COMMAND} inside ${DOCKER_IMG_NAME}..."
# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).
docker run --rm --pid=host \
    -v ${WORKSPACE}:/workspace \
    -w /workspace \
    "${DOCKER_IMG_NAME}" \
    ${COMMAND}
