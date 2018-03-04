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
#
# A script to run multiple GPU tests in parallel controlled with an environment
# variable. This script will assume that when it runs, one of the locks are
# already released. So the program calling this script is expected to make sure
# that only $TF_GPU_COUNT processes are running at any gien time.
#
# Required environment variables:
#     TF_GPU_COUNT = Number of GPUs available. This HAS TO BE IN SYNC with the
#                    value of --local_test_jobs flag for bazel.

BASH_VER_MAJOR=$(echo ${BASH_VERSION} | cut -d '.' -f 1)
BASH_VER_MINOR=$(echo ${BASH_VERSION} | cut -d '.' -f 2)

if [[ ${BASH_VER_MAJOR} -lt 4 ]]; then
  echo "Insufficient bash version: ${BASH_VERSION} < 4.2" >&2
  exit 1
elif [[ ${BASH_VER_MAJOR} -eq 4 ]] && [[ ${BASH_VER_MINOR} -lt 2 ]]; then
  echo "Insufficient bash version: ${BASH_VERSION} < 4.2" >&2
  exit 1
fi

TF_GPU_COUNT=${TF_GPU_COUNT:-8}

for i in `seq 0 $((TF_GPU_COUNT-1))`; do
  exec {lock_fd}>/var/lock/gpulock$i || exit 1
  if flock -n "$lock_fd";
  then
    (
      # This export only works within the brackets, so it is isolated to one
      # single command.
      export CUDA_VISIBLE_DEVICES=$i
      echo "Running test $* on GPU $CUDA_VISIBLE_DEVICES"
      $@
    )
    return_code=$?
    flock -u "$lock_fd"
    exit $return_code
  fi
done

echo "Cannot find a free GPU to run the test $* on, exiting with failure..."
exit 1

