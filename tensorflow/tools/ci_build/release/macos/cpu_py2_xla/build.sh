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
set -e
set -x

source tensorflow/tools/ci_build/release/common.sh
install_bazelisk

# Pick a more recent version of xcode
sudo xcode-select --switch /Applications/Xcode_9.2.app/Contents/Developer

# Set bazel temporary directory to avoid space issues.
export TEST_TMPDIR=/tmpfs/bazel_tmp
mkdir "${TEST_TMPDIR}"
source tensorflow/tools/ci_build/builds/builds_common.sh

# Do the pyenv shim thing to avoid build breakages.
export PATH=$(echo "$PATH" | sed "s#${HOME}/.pyenv/shims:##g")

# Fix numpy version
sudo pip2 install numpy==1.12.1
sudo pip2 install grpcio
sudo pip2 install --upgrade setuptools==39.1.0

py_ver="2"

bazel clean
export TF_ENABLE_XLA=1
export TF_NEED_GCP=1
export TF_NEED_HDFS=1
export TF_NEED_S3=1
export PYTHON_BIN_PATH=$(which python"${py_ver}")

yes "" | "$PYTHON_BIN_PATH" configure.py
tag_filters="-no_oss,-oss_serial,-benchmark-test,-nomac,-no_mac,-no_oss_py2"

bazel test --test_output=errors -c opt \
  --test_tag_filters="${tag_filters}" \
  --build_tag_filters="${tag_filters}" \
  --distinct_host_configuration=false --build_tests_only \
  --java_toolchain=@bazel_tools//tools/jdk:toolchain_hostjdk8 \
  --test_timeout 300,450,1200,3600 -- \
  //tensorflow/... -//tensorflow/compiler/...
