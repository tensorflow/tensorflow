#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# This script assumes the standard setup on tensorflow Jenkins windows machines.
# It is NOT guaranteed to work on any other machine. Use at your own risk!
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Anaconda3
# * Bazel windows executable copied as "bazel.exe" and included in PATH.

# All commands shall pass, and all should be visible.
set -x
set -e

# This script is under <repo_root>/tensorflow/tools/ci_build/windows/cpu/bazel
# Change into repository root.
script_dir=$(dirname $0)
cd ${script_dir%%tensorflow/tools/ci_build/windows/cpu/bazel}.

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/cpu/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# bazel clean --expunge doesn't work on Windows yet.
# Clean the output base manually to ensure build correctness
bazel clean
output_base=$(bazel info output_base)
bazel shutdown
# Sleep 5s to wait for jvm shutdown completely
# otherwise rm will fail with device or resource busy error
sleep 5
rm -rf ${output_base}

export TF_NEED_CUDA=0
echo "" | ./configure

failing_tests="\
    //tensorflow/core:example_example_parser_configuration_test + \
    //tensorflow/core/kernels:sparse_dense_binary_op_shared_test + \
    //tensorflow/core/kernels:sparse_reduce_sum_op_test + \
    //tensorflow/core:lib_core_status_test + \
    //tensorflow/core:lib_monitoring_collection_registry_test + \
    //tensorflow/core:lib_strings_numbers_test + \
    //tensorflow/core:lib_strings_str_util_test + \
    //tensorflow/core/platform/hadoop:hadoop_file_system_test + \
    //tensorflow/core:platform_file_system_test + \
    //tensorflow/core:platform_logging_test + \
    //tensorflow/core:util_sparse_sparse_tensor_test + \
    //tensorflow/cc:framework_gradient_checker_test + \
    //tensorflow/cc:framework_gradients_test + \
    //tensorflow/cc:gradients_array_grad_test + \
    //tensorflow/cc:gradients_math_grad_test + \
    //tensorflow/cc:gradients_nn_grad_test + \
    //tensorflow/cc/saved_model:loader_test
"

broken_tests="\
    //tensorflow/cc:framework_cc_ops_test + \
    //tensorflow/core/platform/cloud:time_util_test + \
    //tensorflow/core/platform/cloud:oauth_client_test + \
    //tensorflow/core/platform/cloud:http_request_test + \
    //tensorflow/core/platform/cloud:google_auth_provider_test + \
    //tensorflow/core/platform/cloud:gcs_file_system_test + \
    //tensorflow/core/kernels/cloud:bigquery_table_accessor_test + \
    //tensorflow/core/kernels/hexagon:quantized_matmul_op_for_hexagon_test + \
    //tensorflow/core/kernels:sparse_add_op_test + \
    //tensorflow/core/kernels:spacetobatch_benchmark_test_gpu + \
    //tensorflow/core/kernels:spacetobatch_benchmark_test + \
    //tensorflow/core/kernels:requantize_op_test + \
    //tensorflow/core/kernels:requantization_range_op_test + \
    //tensorflow/core/kernels:quantized_reshape_op_test + \
    //tensorflow/core/kernels:quantized_pooling_ops_test + \
    //tensorflow/core/kernels:quantized_matmul_op_test + \
    //tensorflow/core/kernels:quantized_conv_ops_test + \
    //tensorflow/core/kernels:quantized_concat_op_test + \
    //tensorflow/core/kernels:quantized_bias_add_op_test + \
    //tensorflow/core/kernels:quantized_batch_norm_op_test + \
    //tensorflow/core/kernels:quantized_activation_ops_test + \
    //tensorflow/core/kernels:quantize_op_test + \
    //tensorflow/core/kernels:quantize_down_and_shrink_range_op_test + \
    //tensorflow/core/kernels:quantize_and_dequantize_op_test_gpu + \
    //tensorflow/core/kernels:quantize_and_dequantize_op_test + \
    //tensorflow/core/kernels:quantization_utils_test + \
    //tensorflow/core/kernels:debug_ops_test + \
    //tensorflow/core/kernels:control_flow_ops_test + \
    //tensorflow/core/kernels:cast_op_test_gpu + \
    //tensorflow/core/kernels:cast_op_test + \
    //tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr_test_gpu + \
    //tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr_test + \
    //tensorflow/core/distributed_runtime/rpc:grpc_tensor_coding_test + \
    //tensorflow/core/distributed_runtime/rpc:grpc_session_test_gpu + \
    //tensorflow/core/distributed_runtime/rpc:grpc_session_test + \
    //tensorflow/core/distributed_runtime/rpc:grpc_channel_test_gpu + \
    //tensorflow/core/distributed_runtime/rpc:grpc_channel_test + \
    //tensorflow/core/distributed_runtime:remote_device_test_gpu + \
    //tensorflow/core/distributed_runtime:remote_device_test + \
    //tensorflow/core/distributed_runtime:executor_test_gpu + \
    //tensorflow/core/distributed_runtime:executor_test + \
    //tensorflow/core/debug:debug_gateway_test + \
    //tensorflow/core/debug:debug_grpc_io_utils_test + \
    //tensorflow/core:util_reporter_test + \
    //tensorflow/core:util_memmapped_file_system_test + \
    //tensorflow/core:platform_subprocess_test + \
    //tensorflow/core:platform_profile_utils_cpu_utils_test + \
    //tensorflow/core:platform_port_test + \
    //tensorflow/core:lib_strings_strcat_test + \
    //tensorflow/core:lib_jpeg_jpeg_mem_unittest + \
    //tensorflow/core:lib_core_notification_test + \
    //tensorflow/core:framework_partial_tensor_shape_test + \
    //tensorflow/core/debug:debug_io_utils_test \
"

exclude_tests="${failing_tests} + ${broken_tests}"

BUILD_OPTS='-c opt --cpu=x64_windows_msvc --host_cpu=x64_windows_msvc --copt=/w --verbose_failures --experimental_ui'

# Find all the passing cc_tests on Windows and store them in a variable
passing_tests=$(bazel query "kind(cc_test, //tensorflow/cc/... + //tensorflow/core/...) - (${exclude_tests})" |
  # We need to strip \r so that the result could be store into a variable under MSYS
  tr '\r' ' ')

bazel test $BUILD_OPTS -k $passing_tests

