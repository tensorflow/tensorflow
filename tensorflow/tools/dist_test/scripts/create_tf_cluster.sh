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
# Create a Kubernetes (k8s) cluster of TensorFlow workers
#
# Usage:
#   create_tf_cluster.sh <num_workers> <num_parameter_servers>
#
# In addition, this script obeys values in the folllowing environment variables:
#   TF_DIST_LOCAL_CLUSTER:        create TensorFlow cluster on local machine
#   TF_DIST_SERVER_DOCKER_IMAGE:  overrides the default docker image to launch
#                                 TensorFlow (GRPC) servers with
#   TF_DIST_GCLOUD_PROJECT:       gcloud project in which the GKE cluster
#                                 will be created (valid only if aforementioned
#                                 TF_DIST_GRPC_SERVER_URL is empty).
#   TF_DIST_GCLOUD_COMPUTE_ZONE:  gcloud compute zone.
#   TF_DIST_CONTAINER_CLUSTER:    name of the GKE cluster
#   TF_DIST_GCLOUD_KEY_FILE:      if non-empty, will override GCLOUD_KEY_FILE
#   TF_DIST_GRPC_PORT:            overrides the default port (2222)
#                                 to run the GRPC servers on

# Configurations
# gcloud operation timeout (steps)
GCLOUD_OP_MAX_STEPS=360

GRPC_PORT=${TF_DIST_GRPC_PORT:-2222}

DEFAULT_GCLOUD_BIN=/var/gcloud/google-cloud-sdk/bin/gcloud
GCLOUD_KEY_FILE=${TF_DIST_GCLOUD_KEY_FILE:-\
"/var/gcloud/secrets/tensorflow-testing.json"}
GCLOUD_PROJECT=${TF_DIST_GCLOUD_PROJECT:-"tensorflow-testing"}

GCLOUD_COMPUTE_ZONE=${TF_DIST_GCLOUD_COMPUTE_ZONE:-"us-central1-f"}
CONTAINER_CLUSTER=${TF_DIST_CONTAINER_CLUSTER:-"test-cluster"}

SERVER_DOCKER_IMAGE=${TF_DIST_SERVER_DOCKER_IMAGE:-\
"tensorflow/tf_grpc_test_server"}

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get utility functions
source "${DIR}/utils.sh"

# Check input arguments
if [[ $# != 2 ]]; then
  die "Usage: $0 <num_workers> <num_parameter_servers>"
fi

NUM_WORKERS=$1
NUM_PARAMETER_SERVERS=$2

# Verify port string
if [[ -z $(echo "${GRPC_PORT}" | grep -E "^[0-9]{1,5}") ]]; then
  die "Invalid GRPC port: \"${GRPC_PORT}\""
fi
echo "GRPC port to be used when creating the k8s TensorFlow cluster: "\
"${GRPC_PORT}"

if [[ -z "${TF_DIST_LOCAL_CLUSTER}" ]] ||
   [[ "${TF_DIST_LOCAL_CLUSTER}" == "0" ]]; then
  IS_LOCAL_CLUSTER="0"
else
  IS_LOCAL_CLUSTER="1"
fi

if [[ ${IS_LOCAL_CLUSTER} == "0" ]]; then
  # Locate gcloud binary path
  GCLOUD_BIN=$(which gcloud)
  if [[ -z "${GCLOUD_BIN}" ]]; then
    GCLOUD_BIN="${DEFAULT_GCLOUD_BIN}"
  fi

  if [[ ! -f "${GCLOUD_BIN}" ]]; then
    die "gcloud binary cannot be found at: ${GCLOUD_BIN}"
  fi
  echo "Path to gcloud binary: ${GCLOUD_BIN}"

  # Path to gcloud service key file
  if [[ ! -f "${GCLOUD_KEY_FILE}" ]]; then
    die "gcloud service account key file cannot be found at: ${GCLOUD_KEY_FILE}"
  fi
  echo "Path to gcloud key file: ${GCLOUD_KEY_FILE}"

  echo "GCLOUD_PROJECT: ${GCLOUD_PROJECT}"
  echo "GCLOUD_COMPUTER_ZONE: ${GCLOUD_COMPUTE_ZONE}"
  echo "CONTAINER_CLUSTER: ${CONTAINER_CLUSTER}"

  # Activate gcloud service account
  "${GCLOUD_BIN}" auth activate-service-account --key-file "${GCLOUD_KEY_FILE}"

  # Set gcloud project
  "${GCLOUD_BIN}" config set project "${GCLOUD_PROJECT}"

  # Set compute zone
  "${GCLOUD_BIN}" config set compute/zone "${GCLOUD_COMPUTE_ZONE}"

  # Set container cluster
  "${GCLOUD_BIN}" config set container/cluster "${CONTAINER_CLUSTER}"

  # Get container cluster credentials
  "${GCLOUD_BIN}" container clusters get-credentials "${CONTAINER_CLUSTER}"
  if [[ $? != "0" ]]; then
    die "FAILED to get credentials for container cluster: ${CONTAINER_CLUSTER}"
  fi

  # If there is any existing tf k8s cluster, delete it first
  "${DIR}/delete_tf_cluster.sh" "${GCLOUD_OP_MAX_STEPS}"
fi

# Path to kubectl binary
KUBECTL_BIN=$(dirname "${GCLOUD_BIN}")/kubectl
if [[ ! -f "${KUBECTL_BIN}" ]]; then
  die "kubectl binary cannot be found at: ${KUBECTL_BIN}"
fi
echo "Path to kubectl binary: ${KUBECTL_BIN}"

# Create yaml file for k8s TensorFlow cluster creation
# Path to the (Python) script for generating k8s yaml file
K8S_GEN_TF_YAML="${DIR}/k8s_tensorflow.py"
if [[ ! -f ${K8S_GEN_TF_YAML} ]]; then
  die "FAILED to find yaml-generating script at: ${K8S_GEN_TF_YAML}"
fi

K8S_YAML="/tmp/k8s_tf_lb.yaml"
rm -f "${K8S_YAML}"

echo ""
echo "Generating k8s cluster yaml config file with the following settings"
echo "  Server docker image: ${SERVER_DOCKER_IMAGE}"
echo "  Number of workers: ${NUM_WORKERS}"
echo "  Number of parameter servers: ${NUM_PARAMETER_SERVERS}"
echo "  GRPC port: ${GRPC_PORT}"
echo ""

${K8S_GEN_TF_YAML} \
    --docker_image "${SERVER_DOCKER_IMAGE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_parameter_servers "${NUM_PARAMETER_SERVERS}" \
    --grpc_port "${GRPC_PORT}" \
    --request_load_balancer=True \
    > "${K8S_YAML}" || \
    die "Generation of the yaml configuration file for k8s cluster FAILED"

if [[ ! -f "${K8S_YAML}" ]]; then
    die "FAILED to generate yaml file for TensorFlow k8s container cluster"
else
    echo "Generated yaml configuration file for k8s TensorFlow cluster: "\
"${K8S_YAML}"
    cat "${K8S_YAML}"
fi

# Create tf k8s container cluster
"${KUBECTL_BIN}" create -f "${K8S_YAML}"

# Wait for external IP of worker services to become available
get_tf_worker_external_ip() {
  # Usage: gen_tf_worker_external_ip <WORKER_INDEX>
  # E.g.,  gen_tf_worker_external_ip 2
  echo $("${KUBECTL_BIN}" get svc | grep "^tf-worker${1}" | \
         awk '{print $3}' | grep -E "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
}

if [[ ${IS_LOCAL_CLUSTER} == "0" ]]; then
  echo "Waiting for external IP of tf-worker0 service to emerge..."
  echo ""

  COUNTER=0
  while true; do
    sleep 1
    ((COUNTER++))
    if [[ $(echo "${COUNTER}>${GCLOUD_OP_MAX_STEPS}" | bc -l) == "1" ]]; then
      die "Reached maximum polling steps while waiting for external IP "\
"of tf-worker0 service to emerge"
    fi

    EXTERN_IPS=""
    WORKER_INDEX=0
    N_AVAILABLE_EXTERNAL_IPS=0
    while true; do
      SVC_EXTERN_IP=$(get_tf_worker_external_ip ${WORKER_INDEX})

      if [[ ! -z "${SVC_EXTERN_IP}" ]]; then
        EXTERN_IPS="${EXTERN_IPS} ${SVC_EXTERN_IP}"

        ((N_AVAILABLE_EXTERNAL_IPS++))
      fi

      ((WORKER_INDEX++))
      if [[ ${WORKER_INDEX} == ${NUM_WORKERS} ]]; then
        break;
      fi
    done

    if [[ ${N_AVAILABLE_EXTERNAL_IPS} == ${NUM_WORKERS} ]]; then
      break;
    fi
  done

  GRPC_SERVER_URLS=""
  for IP in ${EXTERN_IPS}; do
    GRPC_SERVER_URLS="${GRPC_SERVER_URLS} grpc://${IP}:${GRPC_PORT}"
  done
  echo "GRPC URLs of tf-workers: ${GRPC_SERVER_URLS}"

else
  echo "Waiting for tf pods to be all running..."
  echo ""

  COUNTER=0
  while true; do
    sleep 1
    ((COUNTER++))
    if [[ $(echo "${COUNTER}>${GCLOUD_OP_MAX_STEPS}" | bc -l) == "1" ]]; then
      die "Reached maximum polling steps while waiting for all tf pods to "\
"be running in local k8s TensorFlow cluster"
    fi

    PODS_STAT=$(are_all_pods_running "${KUBECTL_BIN}")

    if [[ ${PODS_STAT} == "2" ]]; then
      # Error has occurred
      die "Error(s) occurred while tring to launch tf k8s cluster. "\
"One possible cause is that the Docker image used to launch the cluster is "\
"invalid: \"${SERVER_DOCKER_IMAGE}\""
    fi

    if [[ ${PODS_STAT} == "1" ]]; then
      break
    fi
  done

  # Determine the tf-worker0 docker container id
  WORKER0_ID=$(docker ps | grep "k8s_tf-worker0" | awk '{print $1}')
  echo "WORKER0 Docker container ID: ${WORKER0_ID}"

fi


echo "Cluster setup complete."
