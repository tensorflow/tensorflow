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
# This script invokes dist_mnist.py multiple times concurrently to test the
# TensorFlow's distributed runtime over a Kubernetes (k8s) cluster with the
# grpc pods and service set up.
#
# Usage:
#    dist_mnist_test.sh <worker_grpc_url>
#
# worker_grp_url is the IP address or the GRPC URL of the worker of the main
# worker session, e.g., grpc://1.2.3.4:2222


# Configurations
TIMEOUT=120  # Timeout for MNIST replica sessions

# Helper functions
die() {
  echo $@
  exit 1
}

if [[ $# != 1 ]]; then
  die "Usage: $0 <WORKER_GRPC_URL>"
fi
WORKER_GRPC_URL=$1

# Verify the validity of the GRPC URL
if [[ -z $(echo "${WORKER_GRPC_URL}" | \
    grep -E "^grpc://.+:[0-9]+") ]]; then
  die "Invalid worker GRPC URL: \"${WORKER_GRPC_URL}\""
fi

# Current working directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR=$(dirname "${DIR}")/python

MNIST_REPLICA="${PY_DIR}/mnist_replica.py"

WKR_LOG_PREFIX="/tmp/worker"

# First, download the data from a single process, to avoid race-condition
# during data downloading
timeout ${TIMEOUT} python "${MNIST_REPLICA}" \
    --download_only=True || \
    die "Download-only step of MNIST replica FAILED"

# Run a number of workers in parallel
N_WORKERS=2
INDICES=""
IDX=0
while true; do
  timeout ${TIMEOUT} \
    python "${MNIST_REPLICA}" \
        --worker_grpc_url="${WORKER_GRPC_URL}" \
        --worker_index=${IDX} 2>&1 > \
    "${WKR_LOG_PREFIX}${IDX}.log" &
  # TODO(cais): have each trainer process contact a different worker once
  # supervisor and sync_replicas etc. are all working in OSS TensorFlow.

  INDICES="${INDICES} ${IDX}"

  ((IDX++))
  if [[ $(echo "${IDX}==${N_WORKERS}" | bc -l) == "1" ]]; then
    break
  fi
done

# Function for getting final validation cross entropy from worker log files
get_final_val_xent() {
  echo $(cat $1 | grep "^After.*validation cross entropy = " | \
      awk '{print $NF}')
}

# Poll until all final validation cross entropy values become available or
# operation times out
COUNTER=0
while true; do
  ((COUNTER++))
  if [[ $(echo "${COUNTER}>${TIMEOUT}" | bc -l) == "1" ]]; then
    die "Reached maximum polling steps while polling for final validation "\
"cross entropies from all workers"
  fi

  N_AVAIL=0
  VAL_XENTS=""
  for N in ${INDICES}; do
    VAL_XENT=$(get_final_val_xent "${WKR_LOG_PREFIX}${N}.log")
    if [[ ! -z ${VAL_XENT} ]]; then
      ((N_AVAIL++))
      VAL_XENTS="${VAL_XENTS} ${VAL_XENT}"
    fi
  done

  if [[ "${N_AVAIL}" == "2" ]]; then
    # Print out the content of the log files
    for M in ${INDICES}; do
      echo "==================================================="
      echo "===        Log file from worker ${M}               ==="
      cat "${WKR_LOG_PREFIX}${M}.log"
      echo "==================================================="
      echo ""
    done

    break
  else
    sleep 1
  fi
done

# Sanity check on the validation entropies
# TODO(cais): In addition to this basic sanity check, we could run the training
# with 1 and 2 workers, each for a few times and use scipy.stats to do a t-test
# to verify tha tthe 2-worker training gives significantly lower final cross
# entropy
VAL_XENTS=(${VAL_XENTS})
for N in ${INDICES}; do
  echo "Final validation cross entropy from worker${N}: ${VAL_XENTS[N]}"
  if [[ $(echo "${VAL_XENTS[N]}>0" | bc -l) != "1" ]]; then
      die "Sanity checks on the final validation cross entropy values FAILED"
  fi

done
