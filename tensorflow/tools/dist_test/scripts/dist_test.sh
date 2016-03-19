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
#
# --setup-cluster-only lets the script only set up the k8s container network
#
# This script obeys values in the folllowing environment variables:
#   TF_DIST_GRPC_SERVER_URL:      If it is set to a valid grpc server url (e.g.,
#                                 (grpc://1.2.3.4:2222), the script will bypass
#                                 the cluster setup and teardown processes and
#                                 just use this URL.


# Configurations
NUM_WORKERS=2  # Number of worker container
NUM_PARAMETER_SERVERS=2  # Number of parameter servers

# Helper functions
die() {
  echo $@
  exit 1
}

# gcloud operation timeout (steps)
GCLOUD_OP_MAX_STEPS=240

GRPC_SERVER_URL=${TF_DIST_GRPC_SERVER_URL}

# Report gcloud / GKE parameters
echo "GRPC_SERVER_URL: ${GRPC_SERVER_URL}"

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate path to kubectl binary
TEARDOWN_WHEN_DONE=1
if [[ ! -z "${GRPC_SERVER_URL}" ]]; then
  TEARDOWN_WHEN_DONE=0
  # Verify the validity of the GRPC URL
  if [[ -z $(echo "${GRPC_SERVER_URL}" | \
      grep -E "^grpc://.+:[0-9]+") ]]; then
    die "Invalid GRPC_SERVER_URL: \"${GRPC_SERVER_URL}\""
  else
    echo "The preset GRPC_SERVER_URL appears to be valid: ${GRPC_SERVER_URL}"
    echo "Will bypass the TensorFlow k8s cluster setup and teardown process"
    echo ""
  fi
else
  TMP=$(mktemp)
  "${DIR}/create_tf_cluster.sh" ${NUM_WORKERS} ${NUM_PARAMETER_SERVERS} 2>&1 | \
      tee "${TMP}" || \
      die "Creation of TensorFlow k8s cluster FAILED"

  GRPC_SERVER_URL=$(cat ${TMP} | grep "GRPC URL of tf-worker0: .*" | \
      awk '{print $NF}')
  if [[ -z "${GRPC_SERVER_URL}" ]]; then
    die "FAILED to determine GRPC server URL"
  fi
  rm -f ${TMP}

  if [[ $1 == "--setup-cluster-only" ]]; then
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

echo "Performing distributed MNIST training through grpc session @ "\
"${GRPC_SERVER_URL}..."

"${MNIST_DIST_TEST_BIN}" "${GRPC_SERVER_URL}"

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
