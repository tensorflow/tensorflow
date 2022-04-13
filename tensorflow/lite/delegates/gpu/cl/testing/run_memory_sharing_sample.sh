#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

description="Memory sharing sample:
How to use:
[-h or --help, print instructions]
[-m1 or --model1_path, path to the mobilenet v1 model in .tflite format]
[-m2 or --model2_path, path to the mobilenet v2 model in .tflite format]
[-d or --device, select device](optional, if you have few connected devices)"

model1_path=""
model2_path=""
alias ADB='adb'
host=""

while [[ "$1" != "" ]]; do
  case $1 in
    -m1 | --model1_path)
      shift
      model1_path=$1
      ;;
    -m2 | --model2_path)
      shift
      model2_path=$1
      ;;
    -d | --device)
      shift
      if [[ "$1" == "HOST" ]]
      then
      host="HOST"
      fi
      alias ADB='adb -s '$1''
      ;;
    -h | --help)
      echo "$description"
      exit
      ;;
  esac
  shift
done

if [ "$model1_path" = "" ]
then
echo "No mobilenet v1 model provided."
echo "$description"
exit
fi
if [ "$model2_path" = "" ]
then
echo "No mobilenet v2 model provided."
echo "$description"
exit
fi

SHELL_DIR=$(dirname "$0")
BINARY_NAME=memory_sharing_sample
declare -a BUILD_CONFIG

if [[ "$host" == "HOST" ]]
then

os_name=$(uname -s)
if [[ "$os_name" == "Darwin" ]]; then
BUILD_CONFIG=( --config=darwin_x86_64 -c opt )
else
BUILD_CONFIG=( -c opt )
fi

bazel build "${BUILD_CONFIG[@]}" //"$SHELL_DIR":"$BINARY_NAME"
chmod +x bazel-bin/"$SHELL_DIR"/"$BINARY_NAME"
./bazel-bin/"$SHELL_DIR"/"$BINARY_NAME" "$model1_path" "$model2_path"
exit
fi

model1_name=${model1_path##*/}  # finds last token after '/'
model2_name=${model2_path##*/}  # finds last token after '/'

OPENCL_DIR=/data/local/tmp/memory_sharing_sample/

ADB shell mkdir -p $OPENCL_DIR

ADB push "$model1_path" "$OPENCL_DIR"
ADB push "$model2_path" "$OPENCL_DIR"

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

bazel build "${BUILD_CONFIG[@]}" //$SHELL_DIR:$BINARY_NAME

ADB push bazel-bin/$SHELL_DIR/$BINARY_NAME $OPENCL_DIR

ADB shell chmod +x $OPENCL_DIR/$BINARY_NAME
ADB shell "cd $OPENCL_DIR && ./$BINARY_NAME $model1_name $model2_name"

# clean up files from device
ADB shell rm -rf $OPENCL_DIR
