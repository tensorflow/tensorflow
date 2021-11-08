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

# Update bazel
update_bazel_linux

# Setup virtual environment and install dependencies
setup_venv_ubuntu python3.8

export LD_LIBRARY_PATH="/usr/local/cuda:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/tensorrt/lib"

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

tag_filters="gpu,requires-gpu,-no_gpu,-no_oss,-oss_serial,-no_oss_py38,-no_cuda11"

test +e
bazel test \
  --config=release_gpu_linux \
  --repo_env=PYTHON_BIN_PATH="$(which python)" \
  --build_tag_filters="${tag_filters}" \
  --test_tag_filters="${tag_filters}" \
  --test_lang_filters=py \
  --test_output=errors --verbose_failures=true --keep_going \
  --test_timeout="300,450,1200,3600" --local_test_jobs=4 \
  --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
  -- ${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/...
test_xml_summary_exit

# Remove and cleanup virtual environment
remove_venv_ubuntu
