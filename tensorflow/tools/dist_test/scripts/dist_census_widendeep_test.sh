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
# This script invokes dist_mnist.py multiple times concurrently to test the
# TensorFlow's distributed runtime over a Kubernetes (k8s) cluster with the
# grpc pods and service set up.
#
# Usage:
#    dist_census_widendeep_test.sh <worker_grpc_urls>
#        --num-workers <NUM_WORKERS>
#        --num-parameter-servers <NUM_PARAMETER_SERVERS>
#
# worker_grp_url is the list of IP addresses or the GRPC URLs of the worker of
# the worker sessions, separated with spaces,
# e.g., "grpc://1.2.3.4:2222 grpc://5.6.7.8:2222"
#
# --num-workers <NUM_WORKERS>:
#   Specifies the number of worker pods to use
#
# --num-parameter-server <NUM_PARAMETER_SERVERS>:
#   Specifies the number of parameter servers to use

# Configurations
TIMEOUT=120  # Timeout for MNIST replica sessions

# Helper functions
die() {
  echo $@
  exit 1
}

# Parse command-line arguments
WORKER_GRPC_URLS=$1
shift

# Process additional input arguments
N_WORKERS=2  # Default value
N_PS=2  # Default value
SYNC_REPLICAS=0

while true; do
  if [[ "$1" == "--num-workers" ]]; then
    N_WORKERS=$2
  elif [[ "$1" == "--num-parameter-servers" ]]; then
    N_PS=$2
  elif [[ "$1" == "--sync-replicas" ]]; then
    SYNC_REPLICAS="1"
    die "ERROR: --sync-replicas (synchronized-replicas) mode is not fully "\
"supported by this test yet."
    # TODO(cais): Remove error message once sync-replicas is fully supported
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

echo "N_WORKERS = ${N_WORKERS}"
echo "N_PS = ${N_PS}"

# Dierctory to store the trained model and evaluation results.
# The root (e.g., /shared) must be a directory shared among the workers.
# See volumeMounts fields in k8s_tensorflow.py
MODEL_DIR="/shared/census_widendeep_model"

rm -rf ${MODEL_DIR} || \
    die "Failed to remove existing model directory: ${MODEL_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_PATH="${SCRIPT_DIR}/../python/census_widendeep.py"
if [[ ! -f "${PY_PATH}" ]]; then
  echo "ERROR: Python file does not exist: ${PY_PATH}"
  exit 1
fi

STAGGERED_START_DELAY_SEC=0
WKR_LOG_PREFIX="/tmp/worker_"

IDX=0
LOG_FILES=""
for WORKER_GRPC_URL in ${WORKER_GRPC_URLS}; do
  if [[ ${IDX} != "0" ]]; then
    sleep ${STAGGERED_START_DELAY_SEC}
  fi

  LOG_FILE="${WKR_LOG_PREFIX}${IDX}.log"
  LOG_FILES="${LOG_FILES} ${LOG_FILE}"
  python ${PY_PATH} \
      --master_grpc_url="${WORKER_GRPC_URL}" \
      --num_parameter_servers="${N_PS}" \
      --worker_index="${IDX}" \
      --model_dir="${MODEL_DIR}" \
      --output_dir="/shared/output" \
      --train_steps=1000 \
      --eval_steps=2 2>&1 | tee "${LOG_FILE}" &

  echo "Worker ${IDX}: "
  echo "  GRPC URL: ${WORKER_GRPC_URL}"
  echo "  log file: ${LOG_FILE}"

  ((IDX++))
done

# Wait for all concurrent jobs to finish
wait

# Print logs from the workers
ORD=1
for LOG_FILE in ${LOG_FILES}; do
  echo "==================================================="
  echo "===        Log file from worker ${ORD} / ${N_WORKERS}          ==="
  cat "${LOG_FILE}"
  echo "==================================================="
  echo ""

  ((ORD++))
done

echo "Test for distributed training of Census Wide & Deep model PASSED"