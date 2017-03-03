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
# This script checks for any existing TensorFlow worker services, replication
# controllers and pods in the Kubernetes (k8s) container cluster and delete
# them if there are any.
#
# Usage: delete_tf_cluster [max_steps]
#
# max_steps: Maximum number polling steps for kubectl operations

# Helper functions
die() {
  echo $@
  exit 1
}

# Path to kubectl binary
DEFAULT_KUBECTL_BIN=/var/gcloud/google-cloud-sdk/bin/kubectl
KUBECTL_BIN=$(which kubectl)
if [[ -z "${KUBECTL_BIN}" ]]; then
  KUBECTL_BIN="${DEFAULT_KUBECTL_BIN}"
fi
if [[ ! -f "${KUBECTL_BIN}" ]]; then
  die "kubectl binary cannot be found at: \"${KUBECTL_BIN}\""
else
  echo "Path to kubectl binary: ${KUBECTL_BIN}"
fi

MAX_STEPS=${1:-240}


# Helper functions for kubectl workflow
get_tf_svc_count() {
  echo $("${KUBECTL_BIN}" get svc | grep "tf-" | wc -l)
}

get_tf_rc_count() {
  echo $("${KUBECTL_BIN}" get rc | grep "tf-" | wc -l)
}

get_tf_pods_count() {
  echo $("${KUBECTL_BIN}" get pods | grep "tf-" | wc -l)
}


# Delete all running services, replication-controllers and pods, in that order
ITEMS_TO_DELETE="svc rc pods"
for ITEM in ${ITEMS_TO_DELETE}; do
  K8S_ITEM_COUNT=$(get_tf_${ITEM}_count)
  if [[ ${K8S_ITEM_COUNT} != "0" ]]; then
    echo "There are currently ${K8S_ITEM_COUNT} tf ${ITEM}(s) running. "
    echo "Attempting to delete those..."

    "${KUBECTL_BIN}" delete --all ${ITEM}

    # Wait until all are deleted
    # TODO(cais): Add time out
    COUNTER=0
    while true; do
      sleep 1

      ((COUNTER++))
      if [[ "${COUNTER}" -gt "${MAX_STEPS}" ]]; then
        die "Reached maximum polling steps while trying to delete all tf ${ITEM}"
      fi

      if [[ $(get_tf_${ITEM}_count) == "0" ]]; then
        break
      fi
    done
  fi

done
