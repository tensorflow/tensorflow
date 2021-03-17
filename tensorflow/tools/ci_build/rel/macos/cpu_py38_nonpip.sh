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
export DEVELOPER_DIR=/Applications/Xcode_10.3.app/Contents/Developer
export MACOSX_DEPLOYMENT_TARGET=10.10
sudo xcode-select -s "${DEVELOPER_DIR}"

# Set up and install MacOS pip dependencies.
setup_venv_macos python3.8

# Run configure.
export TF_NEED_CUDA=0
export CC_OPT_FLAGS='-mavx'
export TF2_BEHAVIOR=1
export PYTHON_BIN_PATH=$(which python3.8)
yes "" | "$PYTHON_BIN_PATH" configure.py

tag_filters="-no_oss,-oss_serial,-nomac,-no_mac$(maybe_skip_v1),-gpu,-tpu,-benchmark-test"

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Run tests
bazel test --test_output=errors --config=opt \
  --copt=-DGRPC_BAZEL_BUILD \
  --action_env=TF2_BEHAVIOR="${TF2_BEHAVIOR}" \
  --build_tag_filters="${tag_filters}" \
  --test_tag_filters="${tag_filters}" -- \
  ${DEFAULT_BAZEL_TARGETS} \
  -//tensorflow/lite/...
