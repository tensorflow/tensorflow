#!/usr/bin/env bash

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
# ============================================================================



# Download and build TensorFlow.
set -euxo pipefail
git clone --branch=master --depth=1 https://github.com/tensorflow/tensorflow.git /tensorflow
cd /tensorflow

ln -s $(which ${PYTHON}) /usr/local/bin/python

# Build TensorFlow with support for Intel(R) MKL-DNN
yes "" | ${PYTHON} configure.py && \
 bazel build -c opt --config=mkl --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
    tensorflow/tools/pip_package:build_pip_package && \
 bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
 pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
 rm -rf /tmp/pip && \
 rm -rf /root/.cache


# download and build Horovod
git clone --recursive https://github.com/uber/horovod.git
cd horovod
# export environment
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITH_TENSORFLOW=1
python setup.py sdist
pip --no-cache-dir install --upgrade sdist/horovod*.tar.gz && \
 rm -rf sdist && \
 rm -rf /root/.cache
