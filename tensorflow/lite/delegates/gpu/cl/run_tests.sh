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

shopt -s expand_aliases  # to work with commands aliases in .sh

set -e  # Exit immediately if a command exits with a non-zero status.

description="Script for running tests on android devices
How to use:
[-h or --help, print instructions]
[-t or --test_target, test target]
[-d or --device, select device](optional, if you have few connected devices)"

test_target=""
alias ADB='adb'

while [[ "$1" != "" ]]; do
  case $1 in
    -t | --test_target)
      shift
      test_target=$1
      ;;
    -d | --device)
      shift
      alias ADB='adb -s '$1''
      ;;
    -h | --help)
      echo "$description"
      exit
      ;;
  esac
  shift
done

if [ "$test_target" = "" ]
then
echo "No test target provided."
echo "$description"
exit
fi

OPENCL_DIR=/data/local/tmp/opencl_tests/

cleanup_device() {
  ADB shell rm -rf $OPENCL_DIR
}

ADB shell mkdir -p $OPENCL_DIR
trap "cleanup_device" EXIT

declare -a BUILD_CONFIG
abi_version=$(ADB shell getprop ro.product.cpu.abi | tr -d '\r')
if [[ "$abi_version" == "armeabi-v7a" ]]; then
#"32 bit ARM"
BUILD_CONFIG=( --config=android_arm -c opt --copt=-fPIE --linkopt=-pie )
elif [[ "$abi_version" == "arm64-v8a" ]]; then
#"64 bit ARM"
BUILD_CONFIG=( --config=android_arm64 -c opt )
elif [[ "$abi_version" == "x86_64" ]]; then
# x86_64
BUILD_CONFIG=( --config=android_x86_64 -c opt )
else
echo "Error: Unknown processor ABI"
exit 1
fi

targets=($(bazel query 'tests('$test_target')'))
num_targets=${#targets[@]}
if ((num_targets == 1)); then
  target=${targets[0]}
  executable=${target##*:}  #finds last token after ':'
  bazel build "${BUILD_CONFIG[@]}" $target
  test_path=$(echo $target | tr : /)
  exec_path=bazel-bin/$(echo $test_path | cut -c 3-)
  ADB push "$exec_path" $OPENCL_DIR
  ADB shell chmod +x $OPENCL_DIR/$executable
  ADB shell ./$OPENCL_DIR/$executable
  ADB shell rm -f $OPENCL_DIR/$executable
else # Cleaning log records for multiple test targets
  for ((i = 0; i < num_targets; i++)); do
    echo ${targets[i]}
    target=${targets[i]}
    executable=${target##*:}  #finds last token after ':'
    bazel build "${BUILD_CONFIG[@]}" $target > /dev/null 2>&1
    test_path=$(echo $target | tr : /)
    exec_path=bazel-bin/$(echo $test_path | cut -c 3-)
    ADB push "$exec_path" $OPENCL_DIR > /dev/null 2>&1
    ADB shell chmod +x $OPENCL_DIR/$executable
    ADB shell ./$OPENCL_DIR/$executable --logtostderr 2> /dev/null | grep '\[*\.'
    ADB shell rm -f $OPENCL_DIR/$executable
  done
fi
