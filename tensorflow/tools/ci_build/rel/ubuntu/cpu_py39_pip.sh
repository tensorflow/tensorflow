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
install_bazelisk

# Export required variables for running pip.sh
export OS_TYPE="UBUNTU"
export CONTAINER_TYPE="CPU"
export TF_PYTHON_VERSION='python3.9'
export PYTHON_BIN_PATH="$(which ${TF_PYTHON_VERSION})"

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Export optional variables for running pip.sh
export TF_BUILD_FLAGS="--config=release_cpu_linux"
export TF_TEST_FLAGS="--define=no_tensorflow_py_deps=true --test_lang_filters=py --test_output=errors --verbose_failures=true --keep_going --test_env=TF2_BEHAVIOR=1"
export TF_TEST_TARGETS="${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/... "
export TF_PIP_TESTS="test_pip_virtualenv_non_clean test_pip_virtualenv_clean"
export TF_TEST_FILTER_TAGS='-no_oss,-oss_excluded,-oss_serial,-no_oss_py39,-v1only'
#export IS_NIGHTLY=0 # Not nightly; uncomment if building from tf repo.
export TF_PROJECT_NAME="tensorflow_cpu"
export TF_PIP_TEST_ROOT="pip_test"

./tensorflow/tools/ci_build/builds/pip_new.sh
