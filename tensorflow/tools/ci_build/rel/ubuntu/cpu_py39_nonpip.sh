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

install_ubuntu_16_python_pip_deps python3.9
# Update bazel
install_bazelisk

tag_filters="-no_oss,-oss_serial,-gpu,-tpu,-benchmark-test,-no_oss_py39,-v1only"

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Run tests
set +e
bazel test \
  --config=release_cpu_linux \
  --repo_env=PYTHON_BIN_PATH="$(which python3.9)" \
  --build_tag_filters="${tag_filters}" \
  --test_tag_filters="${tag_filters}" \
  --test_lang_filters=py \
  --test_output=errors \
  --local_test_jobs=8 \
  -- ${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/...
test_xml_summary_exit
