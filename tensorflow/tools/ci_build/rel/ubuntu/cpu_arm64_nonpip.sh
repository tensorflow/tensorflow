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

sudo mkdir /tmpfs
sudo chown ${CI_BUILD_USER}:${CI_BUILD_GROUP} /tmpfs
sudo mkdir /tensorflow
sudo chown ${CI_BUILD_USER}:${CI_BUILD_GROUP} /tensorflow
sudo chown -R ${CI_BUILD_USER}:${CI_BUILD_GROUP} /usr/local/lib/python*
sudo chown ${CI_BUILD_USER}:${CI_BUILD_GROUP} /usr/local/bin
sudo chown -R ${CI_BUILD_USER}:${CI_BUILD_GROUP} /usr/lib/python3/dist-packages

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
export TF_BUILD_FLAGS="--config=mkl_aarch64_threadpool --copt=-flax-vector-conversions"
export TF_TEST_FLAGS="${TF_BUILD_FLAGS} \
    --test_env=TF_ENABLE_ONEDNN_OPTS=1 --test_env=TF2_BEHAVIOR=1 --define=tf_api_version=2 \
    --test_lang_filters=py --flaky_test_attempts=3 --test_size_filters=small,medium \
    --test_output=errors --verbose_failures=true --test_keep_going"
export TF_TEST_TARGETS="${DEFAULT_BAZEL_TARGETS} ${ARM_SKIP_TESTS}"
export TF_FILTER_TAGS="-no_oss,-oss_excluded,-oss_serial,-v1only,-benchmark-test,-no_aarch64,-gpu,-tpu,-no_oss_py38,-no_oss_py39,-no_oss_py310"

if [ ${IS_NIGHTLY} == 1 ]; then
  ./tensorflow/tools/ci_build/update_version.py --nightly
fi

sudo sed -i '/^build --profile/d' /usertools/aarch64.bazelrc
sudo sed -i '\@^build.*=\"/usr/local/bin/python3\"$@d' /usertools/aarch64.bazelrc
sed -i '$ aimport /usertools/aarch64.bazelrc' .bazelrc

bazel test ${TF_TEST_FLAGS} \
    --repo_env=PYTHON_BIN_PATH="$(which python)" \
    --build_tag_filters=${TF_FILTER_TAGS} \
    --test_tag_filters=${TF_FILTER_TAGS} \
    --local_test_jobs=$(grep -c ^processor /proc/cpuinfo) \
    --build_tests_only \
    -- ${TF_TEST_TARGETS}

# Remove virtual environment
remove_venv_ubuntu
