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
# Usage: build_server.sh <docker_image_name> [--test]
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
if [[ $# != 1 ]] && [[ $# != 2 ]]; then
  die "Usage: $0 <docker_image_name> [--test]"
fi

DOCKER_IMG_NAME=$1
shift

# Current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOCKER_FILE="${DIR}/server/Dockerfile"
if [[ $1 == "--test" ]]; then
  DOCKER_FILE="${DIR}/server/Dockerfile.test"
fi
echo "Using Docker file: ${DOCKER_FILE}"

if [[ ! -f "${DOCKER_FILE}" ]]; then
  die "ERROR: Unable to find dockerfile: ${DOCKER_FILE}"
fi
echo "Dockerfile: ${DOCKER_FILE}"

# Call docker build
docker build --no-cache -t "${DOCKER_IMG_NAME}" \
   -f "${DOCKER_FILE}" \
   "${DIR}"
