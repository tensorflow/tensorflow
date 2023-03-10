#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# Set python version string
python_version=$(python3 -c 'import sys; print("python"+str(sys.version_info.major)+"."+str(sys.version_info.minor))')

# Setup virtual environment
setup_venv_ubuntu ${python_version}

# Env vars used to avoid interactive elements of the build.
export HOST_C_COMPILER=(which gcc)
export HOST_CXX_COMPILER=(which g++)
export TF_ENABLE_XLA=1
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_GCP=0
export TF_NEED_S3=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_CUDA=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=1
export TF_NEED_VERBS=0
export TF_NEED_AWS=0
export TF_NEED_GDR=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_KAFKA=0
export TF_NEED_TENSORRT=0

# Export required variables for running the tests
export OS_TYPE="UBUNTU"
export CONTAINER_TYPE="CPU"

# Get the default test targets for bazel
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Get the extended skip test list for arm
source tensorflow/tools/ci_build/build_scripts/ARM_SKIP_TESTS_EXTENDED.sh

# Export optional variables for running the tests
export TF_BUILD_FLAGS="--config=nonccl --config=mkl_aarch64_threadpool \
    --copt=-mtune=generic --copt=-march=armv8-a --copt=-O3 --copt=-flax-vector-conversions"
export TF_TEST_FLAGS="${TF_BUILD_FLAGS} --test_env=TF_ENABLE_ONEDNN_OPTS=1 \
    --test_env=TF2_BEHAVIOR=1 --define=tf_api_version=2 \
    --test_lang_filters=py --flaky_test_attempts=3 --test_size_filters=small,medium"
export TF_TEST_TARGETS="${DEFAULT_BAZEL_TARGETS} ${ARM_SKIP_TESTS}"
export TF_FILTER_TAGS="-no_oss,-oss_serial,-v1only,-benchmark-test,-no_aarch64,-gpu,-tpu,-requires-gpu"

bazel test ${TF_TEST_FLAGS} \
    --repo_env=PYTHON_BIN_PATH="$(which python)" \
    --build_tag_filters=${TF_FILTER_TAGS} \
    --test_tag_filters=${TF_FILTER_TAGS} \
    --local_test_jobs=64 \
    --verbose_failures \
    --build_tests_only \
    -k \
    -- ${TF_TEST_TARGETS}

# Remove virtual environment
remove_venv_ubuntu
