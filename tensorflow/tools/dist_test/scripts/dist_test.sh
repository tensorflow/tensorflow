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
# Performs tests of TensorFlow's distributed runtime over a Kubernetes (k8s)
# container cluster.
#
# This script tears down any existing TensorFlow cluster, consisting of
# services, replication controllers and pods, before creating a new cluster.
# The cluster containers a number of parameter server services and a number of
# worker services. The paramater servers will hold parameters of the ML model,
# e.g., weights and biases of the NN layers, while the workers will hold the
# TensorFlow ops.
#
# Usage:
#   dist_test.sh [--setup-cluster-only]
#                [--num-workers <NUM_WORKERS>]
#                [--num-parameter-servers <NUM_PARAMETER_SERVERS>]
#                [--sync-replicas]
#
# --setup-cluster-only:
#   Lets the script only set up the k8s container network
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
#
# This script obeys values in the folllowing environment variables:
#   TF_DIST_GRPC_SERVER_URLS:     If it is set to a list of valid server urls,
#                                 separated with spaces or commas
#                                 (e.g., "grpc://1.2.3.4:2222 grpc//5.6.7.8:2222"),
#                                 the script will bypass the cluster setup and
#                                 teardown processes and just use this URL.


# Helper functions
die() {
  echo $@
  exit 1
}

# Parse input arguments: number of workers
# Default values:
NUM_WORKERS=2  # Number of worker container
NUM_PARAMETER_SERVERS=2  # Number of parameter servers
SYNC_REPLICAS=0
SETUP_CLUSTER_ONLY=0

while true; do
  if [[ "$1" == "--num-workers" ]]; then
    NUM_WORKERS=$2
  elif [[ "$1" == "--num-parameter-servers" ]]; then
    NUM_PARAMETER_SERVERS=$2
  elif [[ "$1" == "--sync-replicas" ]]; then
    SYNC_REPLICAS=1
  elif [[ "$1" == "--setup-cluster-only" ]]; then
    SETUP_CLUSTER_ONLY=1
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

echo "NUM_WORKERS = ${NUM_WORKERS}"
echo "NUM_PARAMETER_SERVERS = ${NUM_PARAMETER_SERVERS}"
echo "SETUP_CLUSTER_ONLY = ${SETUP_CLUSTER_ONLY}"

# gcloud operation timeout (steps)
GCLOUD_OP_MAX_STEPS=240

if [[ ! -z ${TF_DIST_GRPC_SERVER_URLS} ]]; then
  GRPC_SERVER_URLS=${TF_DIST_GRPC_SERVER_URLS}
  GRPC_SERVER_URLS=$(echo ${GRPC_SERVER_URLS} | sed -e 's/,/ /g')
fi

# Report gcloud / GKE parameters
echo "GRPC_SERVER_URLS: ${GRPC_SERVER_URLS}"
echo "SYNC_REPLICAS: ${SYNC_REPLICAS}"

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate path to kubectl binary
TEARDOWN_WHEN_DONE=1
if [[ ! -z "${GRPC_SERVER_URLS}" ]]; then
  TEARDOWN_WHEN_DONE=0
  # Verify the validity of the GRPC URL
  for GRPC_SERVER_URL in ${GRPC_SEVER_URLS}; do
    if [[ -z $(echo "${GRPC_SERVER_URL}" | \
      grep -E "^grpc://.+:[0-9]+") ]]; then
      die "Invalid GRPC_SERVER_URL: \"${GRPC_SERVER_URL}\""
    fi
  done

  echo "The preset GRPC_SERVER_URLS appears to be valid: ${GRPC_SERVER_URLS}"
  echo "Will bypass the TensorFlow k8s cluster setup and teardown process"
  echo ""

else
  TMP=$(mktemp)
  "${DIR}/create_tf_cluster.sh" ${NUM_WORKERS} ${NUM_PARAMETER_SERVERS} 2>&1 | \
      tee "${TMP}" || \
      die "Creation of TensorFlow k8s cluster FAILED"

  GRPC_SERVER_URLS=$(cat ${TMP} | grep "GRPC URLs of tf-workers: .*" | \
      sed -e 's/GRPC URLs of tf-workers://g')

  if [[ $(echo ${GRPC_SERVER_URLS} | wc -w) != ${NUM_WORKERS} ]]; then
    die "FAILED to determine GRPC server URLs of all workers"
  fi
  rm -f ${TMP}

  if [[ ${SETUP_CLUSTER_ONLY} == "1" ]]; then
    echo "Skipping testing of distributed runtime due to "\
"option flag --setup-cluster-only"
    exit 0
  fi
fi

# Invoke script to perform distributed MNIST training
MNIST_DIST_TEST_BIN="${DIR}/dist_mnist_test.sh"
if [[ ! -f "${MNIST_DIST_TEST_BIN}" ]]; then
  die "FAILED to find distributed mnist client test script at "\
"${MNIST_DIST_TEST_BIN}"
fi

echo "Performing distributed MNIST training through grpc sessions @ "\
"${GRPC_SERVER_URLS}..."

SYNC_REPLICAS_FLAG=""
if [[ ${SYNC_REPLICAS} == "1" ]]; then
  SYNC_REPLICAS_FLAG="--sync-replicas"
fi

"${MNIST_DIST_TEST_BIN}" "${GRPC_SERVER_URLS}" \
    --num-workers "${NUM_WORKERS}" \
    --num-parameter-servers "${NUM_PARAMETER_SERVERS}" \
    ${SYNC_REPLICAS_FLAG}

if [[ $? == "0" ]]; then
  echo "MNIST-replica test PASSED"
else
  die "MNIST-replica test FAILED"
fi

# Tear down current k8s TensorFlow cluster
if [[ "${TEARDOWN_WHEN_DONE}" == "1" ]]; then
  echo "Tearing down k8s TensorFlow cluster..."
  "${DIR}/delete_tf_cluster.sh" "${GCLOUD_OP_MAX_STEPS}" && \
      echo "Cluster tear-down SUCCEEDED" || \
      die "Cluster tear-down FAILED"
fi
echo "SUCCESS: Test of distributed TensorFlow runtime PASSED"
