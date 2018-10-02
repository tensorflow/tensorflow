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
# Tests distributed TensorFlow on a locally running TF GRPC cluster.
#
# This script performs the following steps:
# 1) Build the docker image capable of running distributed TensorFlow in docker.
# 2) Run a container from the aforementioned image and start docker service
#    in it
# 3) Call a script to launch a distributed TensorFlow GRPC cluster inside the container
#    and run the distributed test suite.
#
# Usage: local_test.sh <whl_file_location>
#                      [--leave_container_running]
#                      [--model_name <MODEL_NAME>]
#                      [--num_workers <NUM_WORKERS>]
#                      [--num_parameter_servers <NUM_PARAMETER_SERVERS>]
#                      [--sync_replicas]
#
# E.g., local_test.sh <whl_file_location> --model_name CENSUS_WIDENDEEP
#       local_test.sh <whl_file_location> --num_workers 3 --num_parameter_servers 3
#
# Arguments:
# whl_file_location: URL from which the TensorFlow whl file will be acquired.
#   E.g.: https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.5.0-cp27-none-linux_x86_64.whl
#   E.g.: /path/to/folder/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl
#
# --leave_container_running:  Do not stop the docker-in-docker container after
#                             the termination of the tests, e.g., for debugging
#
# --num_workers <NUM_WORKERS>:
#   Specifies the number of worker pods to start
#
# --num_parameter_server <NUM_PARAMETER_SERVERS>:
#   Specifies the number of parameter servers to start
#
# --sync_replicas
#   Use the synchronized-replica mode. The parameter updates from the replicas
#   (workers) will be aggregated before applied, which avoids stale parameter
#   updates.
#
#
# In addition, this script obeys the following environment variables:
# TF_DIST_DOCKER_NO_CACHE:      do not use cache when building docker images

die() {
  echo $@
  exit 1
}

# Configurations
DOCKER_IMG_NAME="tensorflow/tf-dist-test-local-cluster"

# Parse input arguments
LEAVE_CONTAINER_RUNNING=0
MODEL_NAME=""
MODEL_NAME_FLAG=""
NUM_WORKERS=2
NUM_PARAMETER_SERVERS=2
SYNC_REPLICAS_FLAG=""

WHL_FILE_LOCATION=${1}
if [[ -z "${WHL_FILE_LOCATION}" ]]; then
  echo "WARNING: No wheel url passed. Will use latest tf-nightly cpu p2 wheel."
fi

while true; do
  if [[ $1 == "--leave_container_running" ]]; then
    LEAVE_CONTAINER_RUNNING=1
  elif [[ $1 == "--model_name" ]]; then
    MODEL_NAME="$2"
    MODEL_NAME_FLAG="--model_name ${MODEL_NAME}"
  elif [[ $1 == "--num_workers" ]]; then
    NUM_WORKERS=$2
  elif [[ $1 == "--num_parameter_servers" ]]; then
    NUM_PARAMETER_SERVERS=$2
  elif [[ $1 == "--sync_replicas" ]]; then
    SYNC_REPLICAS_FLAG="--sync_replicas"
  elif [[ $1 == "--WHL_FILE_LOCATION" ]]; then
    WHL_FILE_LOCATION=$2
  fi

  shift
  if [[ -z $1 ]]; then
    break
  fi
done

echo "LEAVE_CONTAINER_RUNNING: ${LEAVE_CONTAINER_RUNNING}"
echo "MODEL_NAME: \"${MODEL_NAME}\""
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "NUM_PARAMETER_SERVERS: ${NUM_PARAMETER_SERVERS}"
echo "SYNC_REPLICAS_FLAG: \"${SYNC_REPLICAS_FLAG}\""

# Current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get utility functions
source ${DIR}/scripts/utils.sh

# Build docker image for local distributed TensorFlow cluster.
NO_CACHE_FLAG=""
if [[ ! -z "${TF_DIST_DOCKER_NO_CACHE}" ]] &&
   [[ "${TF_DIST_DOCKER_NO_CACHE}" != "0" ]]; then
  NO_CACHE_FLAG="--no-cache"
fi

# Create docker build context directory.
BUILD_DIR=$(mktemp -d)
echo ""
echo "Using whl file location: ${WHL_FILE_LOCATION}"
echo "Building in temporary directory: ${BUILD_DIR}"

cp -r ${DIR}/* "${BUILD_DIR}"/ || \
  die "Failed to copy files to ${BUILD_DIR}"

# Download whl file into the build context directory.
if [[ -z "${WHL_FILE_LOCATION}" ]]; then
  pip2 download --no-deps tf-nightly
  cp tf-nightly-*.whl "${BUILD_DIR}"/tensorflow-none-any.whl
elif [[ $WHL_FILE_LOCATION =~ 'http://' || $WHL_FILE_LOCATION =~ 'https://' ]]; then
    # Download whl file into the build context directory.
    wget -P "${BUILD_DIR}" "${WHL_FILE_LOCATION}" || \
        die "Failed to download tensorflow whl file from URL: ${WHL_FILE_LOCATION}"
else
    cp "${WHL_FILE_LOCATION}" "${BUILD_DIR}"
fi

# Build docker image for test.
docker build ${NO_CACHE_FLAG} -t ${DOCKER_IMG_NAME} \
   -f "${BUILD_DIR}/Dockerfile.local" "${BUILD_DIR}" || \
   die "Failed to build docker image: ${DOCKER_IMG_NAME}"

# Clean up docker build context directory.
rm -rf "${BUILD_DIR}"

# Run docker image for test.
docker run ${DOCKER_IMG_NAME} \
    /var/tf_dist_test/scripts/dist_mnist_test.sh \
    --ps_hosts $(seq -f "localhost:%g" -s "," \
                 2000 $((2000 + NUM_PARAMETER_SERVERS - 1))) \
    --worker_hosts $(seq -f "localhost:%g" -s "," \
                     3000 $((3000 + NUM_WORKERS - 1))) \
    --num_gpus 0 ${SYNC_REPLICAS_FLAG}
