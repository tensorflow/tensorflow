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
#    dist_mnist_test.sh <worker_grpc_urls>
#                       [--num-workers <NUM_WORKERS>]
#                       [--num-parameter-servers <NUM_PARAMETER_SERVERS>]
#                       [--sync-replicas]
#
# --sync-replicas
#   Use the synchronized-replica mode. The parameter updates from the replicas
#   (workers) will be aggregated before applied, which avoids stale parameter
#   updates.
#
# worker_grp_url is the list of IP addresses or the GRPC URLs of the worker of
# the worker sessions, separated with spaces,
# e.g., "grpc://1.2.3.4:2222 grpc://5.6.7.8:2222"
#
# --num-workers <NUM_WORKERS>:
#   Specifies the number of workers to run


# Configurations
TIMEOUT=120  # Timeout for MNIST replica sessions

# Helper functions
die() {
  echo $@
  exit 1
}

if [[ $# == "0" ]]; then
  die "Usage: $0 <WORKER_GRPC_URLS> [--num-workers <NUM_WORKERS>] "\
"[--num-parameter-servers <NUM_PARAMETER_SERVERS>] [--sync-replicas]"
fi

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

SYNC_REPLICAS_FLAG=""
if [[ ${SYNC_REPLICAS} == "1" ]]; then
  SYNC_REPLICAS_FLAG="--sync_replicas"
fi

echo "N_WORKERS = ${N_WORKERS}"
echo "N_PS = ${N_PS}"
echo "SYNC_REPLICAS = ${SYNC_REPLICAS}"
echo "SYNC_REPLICAS_FLAG = ${SYNC_REPLICAS_FLAG}"

# Verify the validity of the GRPC URLs
for WORKER_GRPC_URL in ${WORKER_GRPC_URLS}; do
  if [[ -z $(echo "${WORKER_GRPC_URL}" | \
       grep -E "^grpc://.+:[0-9]+") ]]; then
    die "Invalid worker GRPC URL: \"${WORKER_GRPC_URL}\""
  fi
done

# Current working directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR=$(dirname "${DIR}")/python

MNIST_REPLICA="${PY_DIR}/mnist_replica.py"

WKR_LOG_PREFIX="/tmp/worker"

# First, download the data from a single process, to avoid race-condition
# during data downloading
WORKER_GRPC_URL_0=$(echo ${WORKER_GRPC_URLS} | awk '{print $1}')

timeout ${TIMEOUT} python "${MNIST_REPLICA}" \
    --worker_grpc_url="${WORKER_GRPC_URL_0}" \
    --worker_index=0 \
    --num_workers=${N_WORKERS} \
    --num_parameter_servers=${N_PS} \
    ${SYNC_REPLICAS_FLAG} \
    --download_only || \
    die "Download-only step of MNIST replica FAILED"

# Run a number of workers in parallel
echo "${N_WORKERS} worker process(es) running in parallel..."

INDICES=""
IDX=0
URLS=($WORKER_GRPC_URLS)
while true; do
  WORKER_GRPC_URL="${URLS[IDX]}"
  timeout ${TIMEOUT} python "${MNIST_REPLICA}" \
      --worker_grpc_url="${WORKER_GRPC_URL}" \
      --worker_index=${IDX} \
      --num_workers=${N_WORKERS} \
      --num_parameter_servers=${N_PS} \
      ${SYNC_REPLICAS_FLAG} 2>&1 | tee "${WKR_LOG_PREFIX}${IDX}.log" &
  echo "Worker ${IDX}: "
  echo "  GRPC URL: ${WORKER_GRPC_URL}"
  echo "  log file: ${WKR_LOG_PREFIX}${IDX}.log"

  INDICES="${INDICES} ${IDX}"

  ((IDX++))
  if [[ "${IDX}" == "${N_WORKERS}" ]]; then
    break
  fi
done

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
  VAL_XENT=""
  for N in ${INDICES}; do
    if [[ ! -z $(grep "Training ends " "${WKR_LOG_PREFIX}${N}.log") ]]; then
      ((N_AVAIL++))
    fi
  done

  if [[ "${N_AVAIL}" == "${N_WORKERS}" ]]; then
    # Print out the content of the log files
    for M in ${INDICES}; do
      ORD=$(expr ${M} + 1)
      echo "==================================================="
      echo "===        Log file from worker ${ORD} / ${N_WORKERS}          ==="
      cat "${WKR_LOG_PREFIX}${M}.log"
      echo "==================================================="
      echo ""
    done

    break
  else
    sleep 1
  fi
done

# Function for getting final validation cross entropy from worker log files
get_final_val_xent() {
  echo $(cat $1 | grep "^After.*validation cross entropy = " | \
      awk '{print $NF}')
}

VAL_XENT=$(get_final_val_xent "${WKR_LOG_PREFIX}0.log")

# Sanity check on the validation entropies
# TODO(cais): In addition to this basic sanity check, we could run the training
# with 1 and 2 workers, each for a few times and use scipy.stats to do a t-test
# to verify tha tthe 2-worker training gives significantly lower final cross
# entropy
echo "Final validation cross entropy from worker0: ${VAL_XENT}"
if [[ $(echo "${VAL_XENT}>0" | bc -l) != "1" ]]; then
  die "Sanity checks on the final validation cross entropy values FAILED"
fi
