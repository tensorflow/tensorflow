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

# Export required variables for running pip_new.sh
export OS_TYPE="UBUNTU"
export CONTAINER_TYPE="CPU"

${TF_PYTHON_VERSION} -m pip install --upgrade pip wheel
${TF_PYTHON_VERSION} -m pip install --upgrade setuptools
${TF_PYTHON_VERSION} -m pip install -r tensorflow/tools/ci_build/release/requirements_ubuntu.txt
sudo touch /custom_sponge_config.csv
sudo chown ${CI_BUILD_USER}:${CI_BUILD_GROUP} /custom_sponge_config.csv

# Get the default test targets for bazel
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Get the skip test list for arm
source tensorflow/tools/ci_build/build_scripts/ARM_SKIP_TESTS.sh

# Export optional variables for running pip_new.sh
export TF_BUILD_FLAGS="--config=mkl_aarch64_threadpool --copt=-flax-vector-conversions"
export TF_TEST_FLAGS="${TF_BUILD_FLAGS} \
    --test_env=TF_ENABLE_ONEDNN_OPTS=1 --test_env=TF2_BEHAVIOR=1 --define=no_tensorflow_py_deps=true \
    --test_lang_filters=py --flaky_test_attempts=3 --test_size_filters=small,medium \
    --test_output=errors --verbose_failures=true --test_keep_going --notest_verbose_timeout_warnings"
export TF_TEST_TARGETS="${DEFAULT_BAZEL_TARGETS} ${ARM_SKIP_TESTS}"
export TF_PIP_TESTS="test_pip_virtualenv_clean test_pip_virtualenv_oss_serial"
export TF_TEST_FILTER_TAGS="-no_oss,-oss_excluded,-v1only,-benchmark-test,-no_aarch64,-no_oss_py38,-no_oss_py39,-no_oss_py310"
export TF_PIP_TEST_ROOT="pip_test"
export TF_AUDITWHEEL_TARGET_PLAT="manylinux2014"

if [ ${IS_NIGHTLY} == 1 ]; then
  ./tensorflow/tools/ci_build/update_version.py --nightly
fi

sudo sed -i '/^build --profile/d' /usertools/aarch64.bazelrc
sudo sed -i '\@^build.*=\"/usr/local/bin/python3\"$@d' /usertools/aarch64.bazelrc
sed -i '$ aimport /usertools/aarch64.bazelrc' .bazelrc

source tensorflow/tools/ci_build/builds/pip_new.sh

# remove duplicate wheel and copy wheel to mounted volume for local access
rm -rf /tensorflow/pip_test/whl/*linux_aarch64.whl && cp -r /tensorflow/pip_test/whl .
