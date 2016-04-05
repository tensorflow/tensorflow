#!/usr/bin/env bash
# Copyright 2016 Google Inc. All Rights Reserved.
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
# Build and test TensorFlow docker images.
# The tests include Python unit tests-on-install and tutorial tests.
#
# Usage: docker_test.sh <IMAGE_TYPE> <TAG> <WHL_PATH>
# Arguments:
#   IMAGE_TYPE : Type of the image: (CPU|GPU)
#   TAG        : Docker image tag
#   WHL_PATH   : Path to the whl file to be installed inside the docker image
#
#   e.g.: docker_test.sh CPU someone/tensorflow:0.8.0 pip_test/whl/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
#

# Helper functions
# Exit after a failure
die() {
  echo $@
  exit 1
}

# Convert to lower case
to_lower () {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}


# Helper function to traverse directories up until given file is found.
function upsearch () {
  test / == "$PWD" && return || \
      test -e "$1" && echo "$PWD" && return || \
      cd .. && upsearch "$1"
}


# Verify command line argument
if [[ $# != "3" ]]; then
  die "Usage: $(basename $0) <IMAGE_TYPE> <TAG> <WHL_PATH>"
fi
IMAGE_TYPE=$(to_lower "$1")
DOCKER_IMG_TAG=$2
WHL_PATH=$3

# Verify image type
if [[ "${IMAGE_TYPE}" == "cpu" ]]; then
  DOCKERFILE="tensorflow/tools/docker/Dockerfile"
elif [[ "${IMAGE_TYPE}" == "gpu" ]]; then
  DOCKERFILE="tensorflow/tools/docker/Dockerfile.gpu"
else
  die "Unrecognized image type: $1"
fi

# Verify docker binary existence
if [[ -z $(which docker) ]]; then
  die "FAILED: docker binary unavailable"
fi

# Locate the base directory
BASE_DIR=$(upsearch "${DOCKERFILE}")
if [[ -z "${BASE_DIR}" ]]; then
  die "FAILED: Unable to find the base directory where the dockerfile "\
"${DOCKERFFILE} resides"
fi
echo "Base directory: ${BASE_DIR}"

pushd ${BASE_DIR} > /dev/null

# Build docker image
DOCKERFILE_PATH="${BASE_DIR}/${DOCKERFILE}"
DOCKERFILE_DIR="$(dirname ${DOCKERFILE_PATH})"

# Check to make sure that the whl file exists
test -f ${WHL_PATH} || \
    die "whl file does not exist: ${WHL_PATH}"

TMP_WHL_DIR="${DOCKERFILE_DIR}/whl"
mkdir -p "${TMP_WHL_DIR}"
cp "${WHL_PATH}" "${TMP_WHL_DIR}/" || \
    die "FAILED to copy whl file from ${WHL_PATH} to ${TMP_WHL_DIR}/"

docker build -t "${DOCKER_IMG_TAG}" -f "${DOCKERFILE_PATH}" \
"${DOCKERFILE_DIR}" || \
    die "FAILED to build docker image from Dockerfile ${DOCKERFILE_PATH}"

# Clean up
rm -rf "${TMP_WHL_DIR}" || \
    die "Failed to remove temporary directory ${TMP_WHL_DIR}"


# Add extra params for cuda devices and libraries for GPU container.
if [ "${IMAGE_TYPE}" == "gpu" ]; then
  devices=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
  libs=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
  GPU_EXTRA_PARAMS="${devices} ${libs}"
else
  GPU_EXTRA_PARAMS=""
fi

# Run docker image with source directory mapped
docker run -v ${BASE_DIR}:/tensorflow-src -w /tensorflow-src \
${GPU_EXTRA_PARAMS} \
"${DOCKER_IMG_TAG}" \
/bin/bash -c "tensorflow/tools/ci_build/builds/test_installation.sh && "\
"tensorflow/tools/ci_build/builds/test_tutorials.sh && "\
"tensorflow/tools/ci_bukld/builds/integration_tests.sh"

RESULT=$?

popd > /dev/null
if [[ ${RESULT} == 0 ]]; then
  echo "SUCCESS: Built and tested docker image: ${DOCKER_IMG_TAG}"
else
  die "FAILED to build and test docker image: ${DOCKER_IMG_TAG}"
fi
