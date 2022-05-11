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

# Selects a version of Xcode.
export DEVELOPER_DIR=/Applications/Xcode_11.3.app/Contents/Developer
sudo xcode-select -s "${DEVELOPER_DIR}"

# Set up py37 via pyenv and check it worked
PY_VERSION=3.7.9
setup_python_from_pyenv_macos "${PY_VERSION}"
python -m venv .tf-venv && source .tf-venv/bin/activate

# Set up and install MacOS pip dependencies.
install_macos_pip_deps

tag_filters="-no_oss,-oss_serial,-nomac,-no_mac$(maybe_skip_v1),-gpu,-tpu,-benchmark-test"

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Run tests
# Pass PYENV_VERSION since we're using pyenv. See b/182399580
bazel test \
  --config=release_cpu_macos \
  --action_env PYENV_VERSION="${PY_VERSION}" \
  --build_tag_filters="${tag_filters}" \
  --test_tag_filters="${tag_filters}" \
  --test_output=errors \
  -- ${DEFAULT_BAZEL_TARGETS} \
  -//tensorflow/lite/... -//tensorflow/compiler/aot/...
