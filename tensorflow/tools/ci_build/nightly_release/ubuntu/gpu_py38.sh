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

# Setup virtual environment and install dependencies
setup_venv_ubuntu python3.8

export PYTHON_BIN_PATH=$(which python)
"$PYTHON_BIN_PATH" tensorflow/tools/ci_build/update_version.py --nightly

# Build the pip package
bazel build \
  --config=release_gpu_linux \
  --action_env=PYTHON_BIN_PATH="$PYTHON_BIN_PATH" \
  tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package pip_pkg --nightly_flag
./bazel-bin/tensorflow/tools/pip_package/build_pip_package pip_pkg --gpu --nightly_flag

# Upload wheel files
upload_wheel_gpu_ubuntu

# Remove and cleanup virtual environment
remove_venv_ubuntu
