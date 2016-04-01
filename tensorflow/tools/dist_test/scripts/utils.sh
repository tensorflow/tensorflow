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
# Utility functions for dist_test scripts


# Print info and exit with code 1
die() {
  echo $@
  exit 1
}


# Determine if all k8s pods in a namespace are all in the "Running" state
are_all_pods_running() {
  # Usage: are_all_pods_running <KUBECTL_BIN> [namespace]
  KUBECTL_BIN=$1

  if [[ -z "$2" ]]; then
    NS_FLAG=""
  else
    NS_FLAG="--namespace=$2"
  fi

  sleep 1  # Wait for the status to settle
  NPODS=$("${KUBECTL_BIN}" "${NS_FLAG}" get pods | tail -n +2 | wc -l)
  NRUNNING=$("${KUBECTL_BIN}" "${NS_FLAG}" get pods | tail -n +2 | \
      grep "Running" | wc -l)
  NERR=$("${KUBECTL_BIN}" "${NS_FLAG}" get pods | tail -n +2 | \
      grep "Err" | wc -l)

  if [[ ${NERR} != "0" ]]; then
    # "2" signifies that error has occurred
    echo "2"
  elif [[ ${NPODS} == ${NRUNNING} ]]; then
    # "1" signifies that all pods are in Running state
    echo "1"
  else
    # "0" signifies that some pods have not entered Running state, but
    # no error has occurred
    echo "0"
  fi
}
