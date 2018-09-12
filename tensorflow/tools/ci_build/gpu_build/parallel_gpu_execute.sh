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
# variable.
#
# Required environment variables:
#     TF_GPU_COUNT = Number of GPUs available.

TF_GPU_COUNT=${TF_GPU_COUNT:-8}
TF_TESTS_PER_GPU=${TF_TESTS_PER_GPU:-4}
# We want to allow running one of the following configs:
#  - 4 tests per GPU on k80
#  - 8 tests per GPU on p100
# p100 has minimum 12G memory. Therefore, we should limit each test to 1.5G.
# To leave some room in case we want to run more tests in parallel in the
# future and to use a rounder number, we set it to 1G.
export TF_PER_DEVICE_MEMORY_LIMIT_MB=1024

mkdir -p /var/lock
# Try to acquire any of the TF_GPU_COUNT * TF_TESTS_PER_GPU
# slots to run a test at.
#
# Prefer to allocate 1 test per GPU over 4 tests on 1 GPU.
# So, we iterate over TF_TESTS_PER_GPU first.
for j in `seq 0 $((TF_TESTS_PER_GPU-1))`; do
  for i in `seq 0 $((TF_GPU_COUNT-1))`; do
    echo "Trying to lock GPU $i for index $j"
    exec {lock_fd}>/var/lock/gpulock${i}_${j} || exit 1
    if flock -n "$lock_fd";
    then
      (
        # This export only works within the brackets, so it is isolated to one
        # single command.
        export CUDA_VISIBLE_DEVICES=$i
        echo "Running test $@ on GPU $CUDA_VISIBLE_DEVICES"
        $@
      )
      return_code=$?
      flock -u "$lock_fd"
      exit $return_code
    fi
  done
done

echo "Cannot find a free GPU to run the test $* on, exiting with failure..."
exit 1
