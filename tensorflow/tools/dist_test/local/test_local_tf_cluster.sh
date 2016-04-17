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
# Usage: test_local_tf_cluster.sh <NUM_WORKERS> <NUM_PARAMETER_SERVERS>
#                                 [--sync-replicas]
#
# --sync-replicas
#   Use the synchronized-replica mode. The parameter updates from the replicas
#   (workers) will be aggregated before applied, which avoids stale parameter
#   updates.

export GCLOUD_BIN=/usr/local/bin/gcloud
export TF_DIST_LOCAL_CLUSTER=1

# Parse input arguments
if [[ $# == 0 ]] || [[ $# == 1 ]]; then
  echo "Usage: $0 <NUM_WORKERS> <NUM_PARAMETER_SERVERS>"
  exit 1
fi

NUM_WORKERS=$1
NUM_PARAMETER_SERVERS=$2

SYNC_REPLICAS_FLAG=""
if [[ $3 == "--sync-replicas" ]]; then
  SYNC_REPLICAS_FLAG="--sync-replicas"
fi

echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "NUM_PARAMETER_SERVERS: ${NUM_PARAMETER_SERVERS}"
echo "SYNC_REPLICAS_FLAG: ${SYNC_REPLICAS_FLAG}"

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

WORKER_URLS=""
IDX=0
while true; do
  WORKER_URLS="${WORKER_URLS},grpc://tf-worker${IDX}:2222"

  ((IDX++))
  if [[ ${IDX} == ${NUM_WORKERS} ]]; then
    break
  fi
done

echo "Worker URLs: ${WORKER_URLS}"

export TF_DIST_GRPC_SERVER_URLS="${WORKER_URLS}"
GRPC_ENV="TF_DIST_GRPC_SERVER_URLS=${TF_DIST_GRPC_SERVER_URLS}"

CMD="${GRPC_ENV} /var/tf-k8s/scripts/dist_test.sh "\
"--num-workers ${NUM_WORKERS} "\
"--num-parameter-servers ${NUM_PARAMETER_SERVERS} "\
"${SYNC_REPLICAS_FLAG}"

docker exec ${DOCKER_CONTAINER_ID} /bin/bash -c "${CMD}"

if [[ $? != "0" ]]; then
  die "Test of local k8s TensorFlow cluster FAILED"
else
  echo "Test of local k8s TensorFlow cluster PASSED"
fi
