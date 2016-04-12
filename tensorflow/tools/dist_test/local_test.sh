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
# Tests distributed TensorFlow on a locally running TF GRPC cluster.
#
# This script peforms the following steps:
# 1) Build the docker-in-docker (dind) image capable of running docker and
#    Kubernetes (k8s) cluster inside.
# 2) Run a container from the aforementioned image and start docker service
#    in it
# 3) Call a script to launch a k8s TensorFlow GRPC cluster inside the container
#    and run the distributed test suite.
#
# Usage: local_test.sh [--leave-container-running]
#                      [--num-workers <NUM_WORKERS>]
#                      [--num-parameter-servers <NUM_PARAMETER_SERVERS>]
#                      [--sync-replicas]
#
# Arguments:
# --leave-container-running:  Do not stop the docker-in-docker container after
#                             the termination of the tests, e.g., for debugging
#
# --num-workers <NUM_WORKERS>:
#   Specifies the number of worker pods to start
#
# --num-parameter-server <NUM_PARAMETER_SERVERS>:
#   Specifies the number of parameter servers to start
#
# --sync-replicas
#   Use the synchronized-replica mode. The parameter updates from the replicas
#   (workers) will be aggregated before applied, which avoids stale parameter
#   updates.
#
# In addition, this script obeys the following environment variables:
# TF_DIST_SERVER_DOCKER_IMAGE:  overrides the default docker image to launch
#                                 TensorFlow (GRPC) servers with
# TF_DIST_DOCKER_NO_CACHE:      do not use cache when building docker images


# Configurations
DOCKER_IMG_NAME="tensorflow/tf-dist-test-local-cluster"
LOCAL_K8S_CACHE=${HOME}/kubernetes

# Helper function
get_container_id_by_image_name() {
    # Get the id of a container by image name
    # Usage: get_docker_container_id_by_image_name <img_name>

    echo $(docker ps | grep $1 | awk '{print $1}')
}

# Parse input arguments
LEAVE_CONTAINER_RUNNING=0
NUM_WORKERS=2
NUM_PARAMETER_SERVERS=2
SYNC_REPLICAS=0

while true; do
  if [[ $1 == "--leave-container-running" ]]; then
    LEAVE_CONTAINER_RUNNING=1
  elif [[ $1 == "--num-workers" ]]; then
    NUM_WORKERS=$2
  elif [[ $1 == "--num-parameter-servers" ]]; then
    NUM_PARAMETER_SERVERS=$2
  elif [[ $1 == "--sync-replicas" ]]; then
    SYNC_REPLICAS=1
  fi

  shift
  if [[ -z $1 ]]; then
    break
  fi
done

echo "LEAVE_CONTAINER_RUNNING: ${LEAVE_CONTAINER_RUNNING}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "NUM_PARAMETER_SERVERS: ${NUM_PARAMETER_SERVERS}"
echo "SYNC_REPLICAS: ${SYNC_REPLICAS}"

# Current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get utility functions
source ${DIR}/scripts/utils.sh


# First, make sure that no docker-in-docker container of the same image
# is already running
if [[ ! -z $(get_container_id_by_image_name ${DOCKER_IMG_NAME}) ]]; then
    die "It appears that there is already at least one Docker container "\
"of image name ${DOCKER_IMG_NAME} running. Please stop it before trying again"
fi

# Build docker-in-docker image for local k8s cluster
NO_CACHE_FLAG=""
if [[ ! -z "${TF_DIST_DOCKER_NO_CACHE}" ]] &&
   [[ "${TF_DIST_DOCKER_NO_CACHE}" != "0" ]]; then
  NO_CACHE_FLAG="--no-cache"
fi

docker build ${NO_CACHE_FLAG} -t ${DOCKER_IMG_NAME} \
   -f ${DIR}/Dockerfile.local ${DIR}


# Attempt to start the docker container with docker, which will run the k8s
# cluster inside.

# Get current script directory
CONTAINER_START_LOG=$(mktemp --suffix=.log)
echo "Log file for starting cluster container: ${CONTAINER_START_LOG}"
echo ""

${DIR}/local/start_tf_cluster_container.sh \
      ${LOCAL_K8S_CACHE} \
      ${DOCKER_IMG_NAME} | \
    tee ${CONTAINER_START_LOG} &

# Poll start log until the k8s service is started properly or when maximum
# attempt count is reached.
MAX_SERVER_POLLING_ATTEMPTS=600

echo "Waiting for docker-in-docker container for local k8s TensorFlow "\
"cluster to start and launch Kubernetes..."

COUNTER=0
while true; do
  sleep 1

  ((COUNTER++))
  if [[ $(echo "${COUNTER}>=${MAX_SERVER_POLLING_ATTEMPTS}" | bc -l) == "1" ]]; then
    die "Reached maximum number of attempts (${MAX_SERVER_POLLING_ATTEMPTS}) "\
"while waiting for docker-in-docker for local k8s TensorFlow cluster to start"
  fi

  # Check for hitting max attempt while trying to start docker-in-docker
  if [[ $(grep -i "Reached maximum number of attempts" \
                  "${CONTAINER_START_LOG}" | wc -l) == "1" ]]; then
    die "Docker-in-docker container for local k8s TensorFlow cluster "\
"FAILED to start"
  fi

  if [[ $(grep -i "Local Kubernetes cluster is running" \
          "${CONTAINER_START_LOG}" | wc -l) == "1" ]]; then
    break
  fi
done

# Determine the id of the docker-in-docker container
DIND_ID=$(get_container_id_by_image_name ${DOCKER_IMG_NAME})

echo "Docker-in-docker container for local k8s TensorFlow cluster has been "\
"started successfully."
echo "Docker-in-docker container ID: ${DIND_ID}"
echo "Launching k8s tf cluster and tests in container ${DIND_ID} ..."
echo ""

# Launch k8s tf cluster in the docker-in-docker container and perform tests
SYNC_REPLICAS_FLAG=""
if [[ ${SYNC_REPLICAS} == "1" ]]; then
  SYNC_REPLICAS_FLAG="--sync-replicas"
fi

docker exec ${DIND_ID} \
       /var/tf-k8s/local/test_local_tf_cluster.sh \
       ${NUM_WORKERS} ${NUM_PARAMETER_SERVERS} ${SYNC_REPLICAS_FLAG}
TEST_RES=$?

# Tear down: stop docker-in-docker container
if [[ ${LEAVE_CONTAINER_RUNNING} == "0" ]]; then
  echo ""
  echo "Stopping docker-in-docker container ${DIND_ID}"

  docker stop --time=1 ${DIND_ID} || \
      echo "WARNING: Failed to stop container ${DIND_ID} !!"

  echo ""
else
  echo "Will NOT terminate DIND container ${DIND_ID}"
fi

if [[ "${TEST_RES}" != "0" ]]; then
    die "Test of distributed TensorFlow runtime on docker-in-docker local "\
"k8s cluster FAILED"
else
    echo "Test of distributed TensorFlow runtime on docker-in-docker local "\
"k8s cluster PASSED"
fi
