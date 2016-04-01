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
# Launch a Kubernetes (k8s) TensorFlow cluster on the local machine and run
# the distributed test suite.
#
# This script assumes that a TensorFlow cluster is already running on the
# local machine and can be controlled by the "kubectl" binary.
#
# Usage: test_local_tf_cluster.sh
#

export GCLOUD_BIN=/usr/local/bin/gcloud
export TF_DIST_LOCAL_CLUSTER=1

# TODO(cais): Do not hard-code the numbers of workers and ps
NUM_WORKERS=2
NUM_PARAMETER_SERVERS=2

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get utility functions
source "${DIR}/../scripts/utils.sh"

# Wait for the kube-system pods to be running
KUBECTL_BIN=$(which kubectl)
if [[ -z ${KUBECTL_BIN} ]]; then
  die "FAILED to find path to kubectl"
fi

echo "Waiting for kube-system pods to be all running..."
echo ""

MAX_ATTEMPTS=360
COUNTER=0
while true; do
  sleep 1
  ((COUNTER++))
  if [[ $(echo "${COUNTER}>${MAX_ATTEMPTS}" | bc -l) == "1" ]]; then
    die "Reached maximum polling attempts while waiting for all pods in "\
"kube-system to be running in local k8s TensorFlow cluster"
  fi

  if [[ $(are_all_pods_running "${KUBECTL_BIN}" "kube-system") == "1" ]]; then
    break
  fi
done

# Create the local k8s tf cluster
${DIR}/../scripts/create_tf_cluster.sh \
    ${NUM_WORKERS} ${NUM_PARAMETER_SERVERS} | \
    tee /tmp/tf_cluster.log || \
    die "FAILED to create local tf cluster"

DOCKER_CONTAINER_ID=$(cat /tmp/tf_cluster.log | \
    grep "Docker container ID" |
    awk '{print $NF}')
if [[ -z "${DOCKER_CONTAINER_ID}" ]]; then
  die "FAILED to determine worker0 Docker container ID"
fi

export TF_DIST_GRPC_SERVER_URL="grpc://tf-worker0:2222"
GRPC_ENV="TF_DIST_GRPC_SERVER_URL=${TF_DIST_GRPC_SERVER_URL}"

docker exec \
    ${DOCKER_CONTAINER_ID} \
    /bin/bash -c \
    "${GRPC_ENV} /var/tf-k8s/scripts/dist_test.sh"

if [[ $? != "0" ]]; then
  die "Test of local k8s TensorFlow cluster FAILED"
else
  echo "Test of local k8s TensorFlow cluster PASSED"
fi
