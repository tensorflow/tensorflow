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
# Starts a docker-in-docker (dind) container that is capable of running docker
# service and Kubernetes (k8s) cluster inside.
#
# Usage: start_tf_cluster_container.sh <local_k8s_dir> <docker_img_name>
#
# local_k8s_dir:   Kubernetes (k8s) source directory on the host
# docker_img_name: Name of the docker image to start
#
# In addition, this script obeys the following environment variables:
# TF_DIST_SERVER_DOCKER_IMAGE:  overrides the default docker image to launch
#                                 TensorFlow (GRPC) servers with

# Parse input arguments
if [[ $# != "2" ]]; then
  echo "Usage: $0 <host_k8s_dir> <docker_img_name>"
  exit 1
fi

HOST_K8S_DIR=$1
DOCKER_IMG_NAME=$2

# Helper functions
die() {
  echo $@
  exit 1
}

# Maximum number of tries to start the docker container with docker running
# inside
MAX_ATTEMPTS=100

# Map environment variables into the docker-in-docker (dind) container
DOCKER_ENV=""
if [[ ! -z "${TF_DIST_SERVER_DOCKER_IMAGE}" ]]; then
  DOCKER_ENV="-e TF_DIST_SERVER_DOCKER_IMAGE=${TF_DIST_SERVER_DOCKER_IMAGE}"
fi

# Verify that the promisc (promiscuous mode) flag is set on docker0 network
# interface
if [[ -z $(netstat -i | grep "^docker0" | awk '{print $NF}' | grep -o P) ]];
then
  die "FAILED: Cannot proceed with dind k8s container creation because "\
"network interface 'docker0' is not set to promisc on the host."
fi

# Create cache for k8s source
if [[ ! -d ${HOST_K8S_DIR} ]]; then
  umask 000
  mkdir -p ${HOST_K8S_DIR} || die "FAILED to create directory for k8s source"
fi

# Attempt to start docker service in docker container.
# Try multiple times if necessary.
COUNTER=1
while true; do
  ((COUNTER++))
  docker run --rm --net=host --privileged ${DOCKER_ENV} \
      -v ${HOST_K8S_DIR}:/local/kubernetes \
      ${DOCKER_IMG_NAME} \
      /var/tf-k8s/local/start_local_k8s_service.sh

  if [[ $? == "23" ]]; then
    if [[ $(echo "${COUNTER}>=${MAX_ATTEMPTS}" | bc -l) == "1" ]]; then
      echo "Reached maximum number of attempts (${MAX_ATTEMPTS}) "\
"while attempting to start docker-in-docker for local k8s TensorFlow cluster"
      exit 1
    fi

    echo "Docker service failed to start."
    echo "Will make another attempt (#${COUNTER}) to start it..."
    sleep 1
  else
    break
  fi
done
