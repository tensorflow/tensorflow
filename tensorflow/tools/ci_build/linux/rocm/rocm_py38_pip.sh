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

install_ubuntu_16_python_pip_deps python3.8

# Update bazel
#
# The update_bazel_linux function installs bazel in $HOME/bin, but adds /home/kbuilder/bin to the path.
# "/home/kbuilder" does not exist in the ROCm docker container, and hence the "which bazel" cmd at the
# end of that function fails!. So explicitly adding $HOME/bin to the PATH here
export PATH="$HOME/bin:$PATH"

update_bazel_linux

# Export required variables for running pip.sh
export OS_TYPE="UBUNTU"
export CONTAINER_TYPE="ROCM"
export TF_PYTHON_VERSION='python3.8'
export PYTHON_BIN_PATH="$(which ${TF_PYTHON_VERSION})"

# .bazelrc does not have all config options listed within it for ROCm (as it does for others)
# So explicitly calling the configure script here to properly setup the ROCm config options
yes "" | TF_NEED_ROCM=1 ./configure

# Setup some env vars needed by the underlying script(s) for building and testing
export N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
export TF_GPU_COUNT=$(lspci|grep 'controller'|grep 'AMD/ATI'|wc -l)
export TF_TESTS_PER_GPU=1
export N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})


# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# # Export optional variables for running pip.sh
export TF_TEST_FILTER_TAGS='gpu,requires-gpu,-no_gpu,-no_oss,-oss_serial,-no_oss_py38,-no_rocm'
export TF_BUILD_FLAGS="--config=release_base "
export TF_TEST_FLAGS="--test_tag_filters=${TF_TEST_FILTER_TAGS} --build_tag_filters=${TF_TEST_FILTER_TAGS} \
 --test_env=TF2_BEHAVIOR=1 \
 --local_test_jobs=${N_TEST_JOBS} \
 --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
 --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
 --test_timeout 600,900,2400,7200 \
 --build_tests_only \
 --test_output=errors \
 --test_sharding_strategy=disabled \
 --define=no_tensorflow_py_deps=true \
 --test_lang_filters=py \
 --verbose_failures=true \
 --keep_going "
export TF_TEST_TARGETS="${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/... "
export TF_PIP_TESTS="test_pip_virtualenv_non_clean test_pip_virtualenv_clean"
export IS_NIGHTLY=0 # Not nightly; uncomment if building from tf repo.
export TF_PROJECT_NAME="tensorflow_rocm"  # single pip package!
export TF_PIP_TEST_ROOT="pip_test"

# To build both tensorflow and tensorflow-gpu pip packages
export TF_BUILD_BOTH_GPU_PACKAGES=0

./tensorflow/tools/ci_build/builds/pip_new.sh
