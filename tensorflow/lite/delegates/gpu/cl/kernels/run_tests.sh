#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

source gbash.sh || exit 1

set -e  # Exit immediately if a command exits with a non-zero status.

DEFINE_string test_target

gbash::init_google "$@"

_DEVICE="${DEVICE:+-s $DEVICE}"

OPENCL_DIR=/data/local/tmp/opencl_tests/

cleanup_device() {
  adb ${_DEVICE} shell rm -rf $OPENCL_DIR
}

adb ${_DEVICE} shell mkdir -p $OPENCL_DIR
trap "cleanup_device" EXIT

targets=($(bazel query 'tests('${FLAGS_test_target}')'))
num_targets=${#targets[@]}
if ((num_targets == 1)); then
  target=${targets[0]}
  executable=${target##*:}  #finds last token after ':'
  bazel build --config=android_arm64 -c opt $target
  test_path=$(echo $target | tr : /)
  exec_path=bazel-bin/$(echo $test_path | cut -c 3-)
  adb ${_DEVICE} push "$exec_path" $OPENCL_DIR
  adb ${_DEVICE} shell chmod +x $OPENCL_DIR/$executable
  adb ${_DEVICE} shell ./$OPENCL_DIR/$executable
  adb ${_DEVICE} shell rm -f $OPENCL_DIR/$executable
else # Cleaning log records for multiple test targets
  for ((i = 0; i < num_targets; i++)); do
    target=${targets[i]}
    executable=${target##*:}  #finds last token after ':'
    bazel build --config=android_arm64 -c opt $target > /dev/null 2>&1
    test_path=$(echo $target | tr : /)
    exec_path=bazel-bin/$(echo $test_path | cut -c 3-)
    adb ${_DEVICE} push "$exec_path" $OPENCL_DIR > /dev/null 2>&1
    adb ${_DEVICE} shell chmod +x $OPENCL_DIR/$executable
    adb ${_DEVICE} shell ./$OPENCL_DIR/$executable --logtostderr 2> /dev/null | grep '\][[:space:]][a-zA-Z][a-zA-Z0-9_]*\.'
    adb ${_DEVICE} shell rm -f $OPENCL_DIR/$executable
  done
fi
