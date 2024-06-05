#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# A script to build XLA sycl target.
#
# Required input:
#     workspace: the local to do this buidl

if [ $# -lt 1 ];then
  echo "Error: workspace not set."
  exit 1
fi

workspace=$1

if [ -e ${workspace} ];then
  time_stamp=$(date +%s%N)
  echo "Warning: ${workspace} exist."
  workspace=$workspace/$time_stamp
  echo "Will use $workspace as new workspace"
fi

mkdir -p $workspace

xla_path=$workspace/xla
cd $workspace
git clone -b yang/ci https://github.com/Intel-tensorflow/xla xla
bash $xla_path/build_tools/sycl/install_bazel.sh $workspace
bash $xla_path/build_tools/sycl/install_oneapi.sh $workspace install
bash $xla_path/build_tools/sycl/build_xla.sh $workspace
bash $xla_path/build_tools/sycl/clean.sh $workspace

