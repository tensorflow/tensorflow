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
# This is the entry-point script to testing TensorFlow's distributed runtime.
# It builds a docker image with the necessary gcloud and Kubernetes (k8s) tools
# installed, and then execute k8s cluster preparation and distributed TensorFlow
# runs from within a container based on the image.
#
# Usage:
#   remote_test.sh [--setup_cluster_only]
#                  [--num_workers <NUM_WORKERS>]
#                  [--num_parameter_servers <NUM_PARAMETER_SERVERS>]
#                  [--sync_replicas]
#
# Arguments:
#   --setup_cluster_only:
#       Setup the TensorFlow k8s cluster only, and do not perform testing of
#       the distributed runtime.
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
# If any of the following environment variable has non-empty values, it will
# be mapped into the docker container to override the default values (see
# dist_test.sh)
#   TF_DIST_GRPC_SERVER_URL:      URL to an existing TensorFlow GRPC server.
#                                 If set to any non-empty and valid value (e.g.,
#                                 grpc://1.2.3.4:2222), it will cause the test
#                                 to bypass the k8s cluster setup and
#                                 teardown process, and just use the this URL
#                                 as the master session.
#   TF_DIST_GCLOUD_PROJECT:       gcloud project in which the GKE cluster
#                                 will be created (takes effect only if
#                                 TF_DIST_GRPC_SERVER_URL is empty, same below)
#   TF_DIST_GCLOUD_COMPUTE_ZONE:  gcloud compute zone.
#   TF_DIST_CONTAINER_CLUSTER:    name of the GKE cluster
#   TF_DIST_GCLOUD_KEY_FILE:      path to the gloud service JSON key file
#   TF_DIST_GRPC_PORT:            port on which to create the TensorFlow GRPC
#                                 servers
#   TF_DIST_DOCKER_NO_CACHE:      do not use cache when building docker images

DOCKER_IMG_NAME="tensorflow/tf-dist-test-client"

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare environment variables for the docker container
DOCKER_ENV_FLAGS=""
if [[ ! -z "$TF_DIST_GRPC_SERVER_URL" ]]; then
  DOCKER_ENV_FLAGS="${DOCKER_ENV_FLAGS} "\
"-e TF_DIST_GRPC_SERVER_URL=${TF_DIST_GRPC_SERVER_URL}"
fi
if [[ ! -z "$TF_DIST_GCLOUD_PROJECT" ]]; then
  DOCKER_ENV_FLAGS="${DOCKER_ENV_FLAGS} "\
"-e TF_DIST_GCLOUD_PROJECT=${TF_DIST_GCLOUD_PROJECT}"
fi
if [[ ! -z "$TF_DIST_GCLOUD_COMPUTE_ZONE" ]]; then
  DOCKER_ENV_FLAGS="${DOCKER_ENV_FLAGS} "\
"-e TF_DIST_GCLOUD_COMPUTE_ZONE=${TF_DIST_GCLOUD_COMPUTE_ZONE}"
fi
if [[ ! -z "$TF_DIST_CONTAINER_CLUSTER" ]]; then
  DOCKER_ENV_FLAGS="${DOCKER_ENV_FLAGS} "\
"-e TF_DIST_CONTAINER_CLUSTER=${TF_DIST_CONTAINER_CLUSTER}"
fi
if [[ ! -z "$TF_DIST_GRPC_PORT" ]]; then
  DOCKER_ENV_FLAGS="${DOCKER_ENV_FLAGS} "\
"-e TF_DIST_GRPC_PORT=${TF_DIST_GRPC_PORT}"
fi

NO_CACHE_FLAG=""
if [[ ! -z "${TF_DIST_DOCKER_NO_CACHE}" ]] &&
   [[ "${TF_DIST_DOCKER_NO_CACHE}" != "0" ]]; then
  NO_CACHE_FLAG="--no-cache"
fi

docker build ${NO_CACHE_FLAG} \
    -t ${DOCKER_IMG_NAME} -f "${DIR}/Dockerfile" "${DIR}"
KEY_FILE=${TF_DIST_GCLOUD_KEY_FILE:-"${HOME}/gcloud-secrets/tensorflow-testing.json"}

docker run --rm -v ${KEY_FILE}:/var/gcloud/secrets/tensorflow-testing.json \
  ${DOCKER_ENV_FLAGS} \
  ${DOCKER_IMG_NAME} \
  /var/tf-dist-test/scripts/dist_test.sh $@
