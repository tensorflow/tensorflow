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
# C++ tests
failing_cpu_cc_tests="\
    //tensorflow/core/kernels:control_flow_ops_test + \
    //tensorflow/core:example_example_parser_configuration_test + \
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
    //tensorflow/cc/saved_model:loader_test \
"

broken_cpu_cc_tests="\
    //tensorflow/core/kernels/hexagon:graph_transferer_test + \
    //tensorflow/cc:framework_cc_ops_test + \
    //tensorflow/core/platform/cloud:time_util_test + \
    //tensorflow/core/platform/cloud:oauth_client_test + \
    //tensorflow/core/platform/cloud:http_request_test + \
    //tensorflow/core/platform/cloud:google_auth_provider_test + \
    //tensorflow/core/platform/cloud:gcs_file_system_test + \
    //tensorflow/core/kernels/cloud:bigquery_table_accessor_test + \
    //tensorflow/core/kernels/hexagon:quantized_matmul_op_for_hexagon_test + \
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
    //tensorflow/core:lib_jpeg_jpeg_mem_unittest + \
    //tensorflow/core/debug:debug_io_utils_test \
"

# lib_core_threadpool_test is timeout, but it passes when running alone
extra_failing_gpu_cc_tests="\
    //tensorflow/core:lib_core_threadpool_test + \
    //tensorflow/core:cuda_libdevice_path_test + \
    //tensorflow/core:common_runtime_direct_session_test + \
    //tensorflow/core:common_runtime_direct_session_with_tracking_alloc_test + \
    //tensorflow/core:gpu_tracer_test + \
    //tensorflow/core:ops_math_grad_test \
"

exclude_cpu_cc_tests="${failing_cpu_cc_tests} + ${broken_cpu_cc_tests}"

exclude_gpu_cc_tests="${extra_failing_gpu_cc_tests} + ${exclude_cpu_cc_tests}"

# Python tests
# The first argument is the name of the python test direcotry
function get_failing_cpu_py_tests() {
    echo "
    //$1/tensorflow/python/kernel_tests:rnn_test + \
    //$1/tensorflow/python/kernel_tests:sets_test + \
    //$1/tensorflow/python/debug:cli_shared_test + \
    //$1/tensorflow/python/debug:command_parser_test + \
    //$1/tensorflow/python/debug:debug_data_test + \
    //$1/tensorflow/python/debug:debug_utils_test + \
    //$1/tensorflow/python/debug:debugger_cli_common_test + \
    //$1/tensorflow/python/debug:framework_test + \
    //$1/tensorflow/python/debug:local_cli_wrapper_test + \
    //$1/tensorflow/python/debug:tensor_format_test + \
    //$1/tensorflow/python:saver_large_variable_test + \
    //$1/tensorflow/python:session_test + \
    //$1/tensorflow/python:basic_session_run_hooks_test + \
    //$1/tensorflow/python:contrib_test + \
    //$1/tensorflow/python/debug:analyzer_cli_test + \
    //$1/tensorflow/python/debug:curses_ui_test + \
    //$1/tensorflow/python/debug:session_debug_file_test + \
    //$1/tensorflow/python/debug:stepper_test + \
    //$1/tensorflow/python:dequantize_op_test + \
    //$1/tensorflow/python:directory_watcher_test + \
    //$1/tensorflow/python:event_multiplexer_test + \
    //$1/tensorflow/python:file_io_test + \
    //$1/tensorflow/python:framework_meta_graph_test + \
    //$1/tensorflow/python:framework_ops_test + \
    //$1/tensorflow/python:framework_tensor_util_test + \
    //$1/tensorflow/python:framework_test_util_test + \
    //$1/tensorflow/python:image_ops_test + \
    //$1/tensorflow/python/kernel_tests:as_string_op_test + \
    //$1/tensorflow/python/kernel_tests:benchmark_test + \
    //$1/tensorflow/python/kernel_tests:cast_op_test + \
    //$1/tensorflow/python/kernel_tests:clip_ops_test + \
    //$1/tensorflow/python/kernel_tests:conv_ops_test + \
    //$1/tensorflow/python/kernel_tests:decode_image_op_test + \
    //$1/tensorflow/python/kernel_tests:depthwise_conv_op_test + \
    //$1/tensorflow/python/kernel_tests:functional_ops_test + \
    //$1/tensorflow/python/kernel_tests:py_func_test + \
    //$1/tensorflow/python/kernel_tests:sparse_matmul_op_test + \
    //$1/tensorflow/python/kernel_tests:string_to_number_op_test + \
    //$1/tensorflow/python/kernel_tests:summary_ops_test + \
    //$1/tensorflow/python/kernel_tests:variable_scope_test + \
    //$1/tensorflow/python:monitored_session_test + \
    //$1/tensorflow/python:nn_batchnorm_test + \
    //$1/tensorflow/python:protobuf_compare_test + \
    //$1/tensorflow/python:quantized_conv_ops_test + \
    //$1/tensorflow/python:saver_test + \
    //$1/tensorflow/python:file_system_test \
    "
}

function get_failing_gpu_py_tests() {
    echo "
    //$1/tensorflow/python/kernel_tests:rnn_test + \
    //$1/tensorflow/python/kernel_tests:sets_test + \
    //$1/tensorflow/python/kernel_tests:diag_op_test + \
    //$1/tensorflow/python/kernel_tests:one_hot_op_test + \
    //$1/tensorflow/python/kernel_tests:trace_op_test + \
    $(get_failing_cpu_py_tests $1)
    "
}

function clean_output_base() {
  # TODO(pcloudy): bazel clean --expunge doesn't work on Windows yet.
  # Clean the output base manually to ensure build correctness
  bazel clean
  output_base=$(bazel info output_base)
  bazel shutdown
  # Sleep 5s to wait for jvm shutdown completely
  # otherwise rm will fail with device or resource busy error
  sleep 5
  rm -rf ${output_base}
}

function run_configure_for_cpu_build {
  export TF_NEED_CUDA=0
  echo "" | ./configure
}

function run_configure_for_gpu_build {
  # Due to a bug in Bazel: https://github.com/bazelbuild/bazel/issues/2182
  # yes "" | ./configure doesn't work on Windows, so we set all the
  # environment variables in advance to avoid interact with the script.
  export TF_NEED_CUDA=1
  export TF_CUDA_VERSION=8.0
  export CUDA_TOOLKIT_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0"
  export TF_CUDNN_VERSION=5
  export CUDNN_INSTALL_PATH="C:/tools/cuda"
  export TF_CUDA_COMPUTE_CAPABILITIES="3.5,5.2"
  echo "" | ./configure
}

function create_python_test_dir() {
  rm -rf "$1"
  mkdir -p "$1"
  cmd /c "mklink /J $1\\tensorflow .\\tensorflow"
}

function reinstall_tensorflow_pip() {
  echo "y" | pip uninstall tensorflow -q || true
  pip install ${1}
}
