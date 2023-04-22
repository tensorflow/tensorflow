#!/usr/bin/env bash

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================

set -ex
git clone --branch=master --depth=1 https://github.com/tensorflow/tensorflow.git /tensorflow || true
cd /tensorflow
ln -snf $(which ${PYTHON}) /usr/local/bin/python
# Run configure.
export TF_NEED_GCP=1
export TF_NEED_HDFS=1
export TF_NEED_S3=1
export TF_NEED_CUDA=0
# TensorRT build failing as of 2019-12-18, see
# https://github.com/tensorflow/tensorflow/issues/35115
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=$(which python3.7)
export TMP=/tmp
yes "" | /usr/local/bin/python configure.py

# Build the pip package and install it
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=opt --config=v2 tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip_pkg --cpu --nightly_flag
ls -al /tmp/pip_pkg
pip --no-cache-dir install --upgrade /tmp/pip_pkg/*.whl
