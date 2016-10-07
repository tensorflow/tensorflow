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
# Performs tests of TensorFlow's distributed runtime over a Kubernetes (k8s)
# container cluster.
#
# This script tears down any existing TensorFlow cluster, consisting of
# services, replication controllers and pods, before creating a new cluster.
# The cluster containers a number of parameter server services and a number of
# worker services. The parameter servers will hold parameters of the ML model,
# e.g., weights and biases of the NN layers, while the workers will hold the
# TensorFlow ops.
#
# Usage:
#   dist_test.sh [--setup_cluster_only]
#                [--model_name (MNIST | CENSUS_WIDENDEEP)]
#                [--num_workers <NUM_WORKERS>]
#                [--num_parameter_servers <NUM_PARAMETER_SERVERS>]
#                [--sync_replicas]
#
# --setup_cluster_only:
#   Lets the script only set up the k8s container network
#
# --model_name
#   Name of the model to test. Default is MNIST.
#
# --num-workers <NUM_WORKERS>:
#   Specifies the number of worker pods to start
#
# --num_parameter_servers <NUM_PARAMETER_SERVERS>:
#   Specifies the number of parameter servers to start
#
# --sync_replicas
#   Use the synchronized-replica mode. The parameter updates from the replicas
#   (workers) will be aggregated before applied, which avoids stale parameter
#   updates.
#
#
# This script obeys values in the following environment variables:
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
MODEL_NAME="MNIST"  # Model name, default is "MNIST"
NUM_WORKERS=2  # Number of worker container
NUM_PARAMETER_SERVERS=2  # Number of parameter servers
SYNC_REPLICAS=0
SETUP_CLUSTER_ONLY=0

while true; do
  if [[ "$1" == "--model_name" ]]; then
    MODEL_NAME=$2
  elif [[ "$1" == "--num_workers" ]]; then
    NUM_WORKERS=$2
  elif [[ "$1" == "--num_parameter_servers" ]]; then
    NUM_PARAMETER_SERVERS=$2
  elif [[ "$1" == "--sync_replicas" ]]; then
    SYNC_REPLICAS=1
  elif [[ "$1" == "--setup_cluster_only" ]]; then
    SETUP_CLUSTER_ONLY=1
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

echo "MODEL_NAME = \"MODEL_NAME\""
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
  for GRPC_SERVER_URL in ${GRPC_SERVER_URLS}; do
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

  GRPC_SERVER_URLS=$(cat ${TMP} | grep "GRPC URLs of tf-worker instances: .*" | \
      sed -e 's/GRPC URLs of tf-worker instances://g')

  GRPC_PS_URLS=$(cat ${TMP} | grep "GRPC URLs of tf-ps instances: .*" | \
      sed -e 's/GRPC URLs of tf-ps instances://g')

  if [[ $(echo ${GRPC_SERVER_URLS} | wc -w) != ${NUM_WORKERS} ]]; then
    die "FAILED to determine GRPC server URLs of all workers"
  fi
  if [[ $(echo ${GRPC_PS_URLS} | wc -w) != ${NUM_PARAMETER_SERVERS} ]]; then
    die "FAILED to determine GRPC server URLs of all parameter servers"
  fi

  WORKER_HOSTS=$(echo "${GRPC_SERVER_URLS}" | sed -e 's/^[[:space:]]*//' | \
                 sed -e 's/grpc:\/\///g' | sed -e 's/ /,/g')
  PS_HOSTS=$(echo "${GRPC_PS_URLS}" | sed -e 's/^[[:space:]]*//' | \
             sed -e 's/grpc:\/\///g' | sed -e 's/ /,/g')

  echo "WORKER_HOSTS = ${WORKER_HOSTS}"
  echo "PS_HOSTS = ${PS_HOSTS}"

  rm -f ${TMP}

  if [[ ${SETUP_CLUSTER_ONLY} == "1" ]]; then
    echo "Skipping testing of distributed runtime due to "\
"option flag --setup_cluster_only"
    exit 0
  fi
fi


# Test routine for model "MNIST"
test_MNIST() {
  # Invoke script to perform distributed MNIST training
  MNIST_DIST_TEST_BIN="${DIR}/dist_mnist_test.sh"
  if [[ ! -f "${MNIST_DIST_TEST_BIN}" ]]; then
    echo "FAILED to find distributed mnist client test script at "\
  "${MNIST_DIST_TEST_BIN}"
    return 1
  fi

  echo "Performing distributed MNIST training through worker grpc sessions @ "\
  "${GRPC_SERVER_URLS}..."

  echo "and ps grpc sessions @ ${GRPC_PS_URLS}"

  SYNC_REPLICAS_FLAG=""
  if [[ ${SYNC_REPLICAS} == "1" ]]; then
    SYNC_REPLICAS_FLAG="--sync_replicas"
  fi

  "${MNIST_DIST_TEST_BIN}" \
      --existing_servers True \
      --ps_hosts "${PS_HOSTS}" \
      --worker_hosts "${WORKER_HOSTS}" \
      --num_gpus 0 \
      ${SYNC_REPLICAS_FLAG}

  if [[ $? == "0" ]]; then
    echo "MNIST-replica test PASSED\n"
  else
    echo "MNIST-replica test FAILED\n"
    return 1
  fi
}

# Test routine for model "CENSUS_WIDENDEEP"
test_CENSUS_WIDENDEEP() {
  # Invoke script to perform distributed census_widendeep training
  CENSUS_WIDENDEEP_DIST_TEST_BIN="${DIR}/dist_census_widendeep_test.sh"
  if [[ ! -f "${CENSUS_WIDENDEEP_DIST_TEST_BIN}" ]]; then
    echo "FAILED to find distributed widen&deep client test script at "\
  "${CENSUS_WIDENDEEP_DIST_TEST_BIN}"
    return 1
  fi

  echo "Performing distributed wide&deep (census) training through grpc "\
  "sessions @ ${GRPC_SERVER_URLS}..."

  "${CENSUS_WIDENDEEP_DIST_TEST_BIN}" "${GRPC_SERVER_URLS}" \
      --num-workers "${NUM_WORKERS}" \
      --num-parameter-servers "${NUM_PARAMETER_SERVERS}"

  if [[ $? == "0" ]]; then
    echo "Census Wide & Deep test PASSED"
    echo ""
  else
    echo "Census Wide & Deep test FAILED"
    echo ""
    return 1
  fi
}

# Validate model name
if [[ $(type -t "test_${MODEL_NAME}") != "function" ]]; then
  die "ERROR: Unsupported model: \"${MODEL_NAME}\""
fi

# Invoke test routine according to model name
"test_${MODEL_NAME}" || \
    die "Test of distributed training of model ${MODEL_NAME} FAILED"

# Tear down current k8s TensorFlow cluster
if [[ "${TEARDOWN_WHEN_DONE}" == "1" ]]; then
  echo "Tearing down k8s TensorFlow cluster..."
  "${DIR}/delete_tf_cluster.sh" "${GCLOUD_OP_MAX_STEPS}" && \
      echo "Cluster tear-down SUCCEEDED" || \
      die "Cluster tear-down FAILED"
fi

echo "SUCCESS: Test of distributed TensorFlow runtime PASSED"
echo ""