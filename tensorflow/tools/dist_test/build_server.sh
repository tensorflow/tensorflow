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
#
# Builds the test server for distributed (GRPC) TensorFlow
#
# Usage: build_server.sh <docker_image_name> <whl_file_location> [--test]
#
# Arguments:
#   docker_image_name: Name of the docker image to build.
#     E.g.: tensorflow/tf_grpc_test_server:0.11.0rc1
#
#   whl_file_location: URL from which the TensorFlow whl file will be downloaded.
#     E.g.: https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl
#     E.g.: /path/to/folder/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl
#
# The optional flag --test lets the script to use the Dockerfile for the
# testing GRPC server. Without the flag, the script will build the non-test
# GRPC server.
#
# Note that the Dockerfile is located in ./server/ but the docker build should
# use the current directory as the context.


# Helper functions
die() {
  echo $@
  exit 1
}

# Check arguments
if [[ $# -lt 2 ]]; then
  die "Usage: $0 <docker_image_name> <whl_location> [--test]"
fi

DOCKER_IMG_NAME=$1
WHL_FILE_LOCATION=$2
shift 2

# Current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR=$(mktemp -d)
echo ""
echo "Using whl file URL: ${WHL_FILE_LOCATION}"
echo "Building in temporary directory: ${BUILD_DIR}"

cp -r ${DIR}/* "${BUILD_DIR}"/ || \
    die "Failed to copy files to ${BUILD_DIR}"

DOCKER_FILE="${BUILD_DIR}/server/Dockerfile"
if [[ $1 == "--test" ]]; then
  DOCKER_FILE="${BUILD_DIR}/server/Dockerfile.test"
fi
echo "Using Docker file: ${DOCKER_FILE}"

if [[ $WHL_FILE_LOCATION =~ 'http://' || $WHL_FILE_LOCATION =~ 'https://' ]]; then
    # Download whl file into the build context directory.
    wget -P "${BUILD_DIR}" "${WHL_FILE_LOCATION}" || \
        die "Failed to download tensorflow whl file from URL: ${WHL_FILE_LOCATION}"
else
    cp "${WHL_FILE_LOCATION}" "${BUILD_DIR}"
fi

# Download whl file into the build context directory.

if [[ ! -f "${DOCKER_FILE}" ]]; then
  die "ERROR: Unable to find dockerfile: ${DOCKER_FILE}"
fi
echo "Dockerfile: ${DOCKER_FILE}"

# Call docker build
docker build --no-cache -t "${DOCKER_IMG_NAME}" \
   -f "${DOCKER_FILE}" "${BUILD_DIR}" || \
   die "Failed to build docker image: ${DOCKER_IMG_NAME}"

# Clean up docker build context directory.
rm -rf "${BUILD_DIR}"
