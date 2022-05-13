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

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Set python version string
py_ver=$(python -c 'import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))')

# Export optional variables for running pip_new.sh
export TF_BUILD_FLAGS="--config=mkl_aarch64 --copt=-mtune=generic --copt=-march=armv8-a \
    --copt=-O3 --copt=-fopenmp --copt=-flax-vector-conversions --linkopt=-lgomp"
export TF_TEST_FLAGS="${TF_BUILD_FLAGS} \
    --test_env=TF_ENABLE_ONEDNN_OPTS=1 --test_env=TF2_BEHAVIOR=1 --test_lang_filters=py \
    --define=no_tensorflow_py_deps=true --verbose_failures=true --test_keep_going"
export TF_TEST_TARGETS="${DEFAULT_BAZEL_TARGETS} \
    -//tensorflow/lite/... \
    -//tensorflow/compiler/mlir/lite/tests:const-fold.mlir.test \
    -//tensorflow/compiler/mlir/lite/tests:prepare-tf.mlir.test \
    -//tensorflow/python:nn_grad_test \
    -//tensorflow/python/eager:forwardprop_test \
    -//tensorflow/python/framework:node_file_writer_test \
    -//tensorflow/python/grappler:memory_optimizer_test \
    -//tensorflow/python/keras/engine:training_arrays_test \
    -//tensorflow/python/kernel_tests/linalg:linear_operator_householder_test \
    -//tensorflow/python/kernel_tests/linalg:linear_operator_inversion_test \
    -//tensorflow/python/kernel_tests/linalg:linear_operator_block_diag_test \
    -//tensorflow/python/kernel_tests/linalg:linear_operator_block_lower_triangular_test \
    -//tensorflow/python/kernel_tests/linalg:linear_operator_kronecker_test \
    -//tensorflow/python/kernel_tests/math_ops:batch_matmul_op_test \
    -//tensorflow/python/kernel_tests/nn_ops:conv_ops_test \
    -//tensorflow/python/kernel_tests/nn_ops:conv2d_backprop_filter_grad_test \
    -//tensorflow/python/kernel_tests/nn_ops:conv3d_backprop_filter_v2_grad_test \
    -//tensorflow/python/kernel_tests/nn_ops:atrous_conv2d_test \
    -//tensorflow/python/ops/parallel_for:math_test"
export TF_PIP_TESTS="test_pip_virtualenv_clean"
export TF_TEST_FILTER_TAGS="-no_oss,-oss_serial,-no_oss_py${py_ver},-gpu,-tpu,-benchmark-test,-v1only,-no_aarch64,-requires-gpu"
export IS_NIGHTLY=0 # Not nightly; uncomment if building from tf repo.
export TF_PROJECT_NAME="tensorflow_cpu"
export TF_PIP_TEST_ROOT="pip_test"
export TF_AUDITWHEEL_TARGET_PLAT="manylinux2014"

source tensorflow/tools/ci_build/builds/pip_new.sh

# remove duplicate wheel and copy wheel to mounted volume for local access
rm -rf /tensorflow/pip_test/whl/*linux_aarch64.whl && cp -r /tensorflow/pip_test/whl .
