#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

set -e

on_device_dir="/data/local/tmp/hexagon_delegate_test/"
hexagon_libs_path=""

if [ "$1" != "" ]; then
  hexagon_libs_path=$1
fi

hexagon_libs_path="${hexagon_libs_path}/libhexagon_nn_skel*"

adb shell rm -rf "${on_device_dir}"
adb shell mkdir "${on_device_dir}"

bazel --bazelrc=/dev/null build -c opt --config=android_arm64 //tensorflow/lite/experimental/delegates/hexagon/builders/tests:all
bazel --bazelrc=/dev/null build -c opt --config=android_arm64 //tensorflow/lite/experimental/delegates/hexagon/hexagon_nn:libhexagon_interface.so

adb push bazel-bin/tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/libhexagon_interface.so "${on_device_dir}"
adb push ${hexagon_libs_path} "${on_device_dir}"

for test_binary in bazel-bin/tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_*_test; do
  echo "Copying $test_binary"
  adb push $test_binary "${on_device_dir}"
  IFS='/'
  read -ra split_path <<< "$test_binary"
  binary_name=${split_path[-1]}
  run_command="/data/local/tmp/hexagon_delegate_test/${binary_name}"
  echo "Running ${run_command}"
  result=$(adb shell 'LD_LIBRARY_PATH=/data/local/tmp/hexagon_delegate_test:${LD_LIBRARY_PATH} '"${run_command}")
  echo 'Output: '
  echo "${result}"
  IFS=$'\n'
  result=($result)
  echo "${result[-1]}"
  if [[ "${result[-1]}" == *"FAILED"* ]]; then
    echo "TEST FAILED"
    exit
  fi
  # Reset delimiter
  IFS=' '
done

echo 'ALL TESTS PASSED -- Yay!!'
