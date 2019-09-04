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

description="Performance profiling sample:
Shows detailed per-layer time in ms for .tflite model.
Uses OpenCL gpu events for measuring.
Works good for Snapdragon(Adreno), Nvidia, Mali. For PowerVR the measurement results are not as precise.
After detailed per-layer timing it measures model execution with CPU timer.
Warning! CPU time can be much slower then time in real application on some devices on Android.
This is because in Android, Activity-based processes can have higher priorities than binary processes.
How to use:
[-h or --help, print instructions]
[-m or --model_path, path to the model in .tflite format]
[-d or --device, select device](optional, if you have few connected devices)"

model_path=""
alias ADB='adb'

while [[ "$1" != "" ]]; do
  case $1 in
    -m | --model_path)
      shift
      model_path=$1
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

if [ "$model_path" = "" ]
then
echo "No model provided."
echo "$description"
exit
fi

SHELL_DIR=$(dirname "$0")

model_name=${model_path##*/}  # finds last token after '/'

declare OPENCL_DIR=/data/local/tmp/profiling_inference/
declare BINARY_NAME=performance_profiling

ADB shell mkdir -p $OPENCL_DIR

ADB push "$model_path" "$OPENCL_DIR"

# push executables and data files to device
# bazel build --config=android_arm -c opt --copt=-fPIE --linkopt=-pie //$SHELL_DIR:$BINARY_NAME  # for 32bit version
bazel build --config=android_arm64 -c opt //$SHELL_DIR:$BINARY_NAME
ADB push bazel-bin/$SHELL_DIR/$BINARY_NAME $OPENCL_DIR

ADB shell chmod +x $OPENCL_DIR/$BINARY_NAME
ADB shell "cd $OPENCL_DIR && ./$BINARY_NAME $model_name"

# clean up files from device
ADB shell rm -rf $OPENCL_DIR
