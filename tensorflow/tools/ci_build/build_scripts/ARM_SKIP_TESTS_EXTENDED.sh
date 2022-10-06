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
#!/bin/bash
set -x

source tensorflow/tools/ci_build/build_scripts/ARM_SKIP_TESTS.sh

ARM_SKIP_TESTS="${ARM_SKIP_TESTS} \
-//tensorflow/compiler/mlir/lite/tests:const-fold.mlir.test \
-//tensorflow/compiler/mlir/tensorflow/tests/... \
-//tensorflow/compiler/mlir/tfr/examples/mnist:mnist_ops_test \
-//tensorflow/compiler/tests:conv2d_test_cpu \
-//tensorflow/compiler/tests:conv2d_test_cpu_mlir_bridge_test \
-//tensorflow/compiler/tests:depthwise_conv_op_test_cpu \
-//tensorflow/compiler/tests:depthwise_conv_op_test_cpu_mlir_bridge_test \
-//tensorflow/compiler/tests:jit_test_cpu \
-//tensorflow/compiler/xla/service/cpu/tests:cpu_eigen_dot_operation_test \
-//tensorflow/compiler/xla/tests:conv_depthwise_backprop_filter_test_cpu \
-//tensorflow/compiler/xla/tests:convolution_dimension_numbers_test_cpu \
-//tensorflow/compiler/xla/tests:convolution_test_cpu \
-//tensorflow/compiler/xla/tests:convolution_variants_test_cpu \
-//tensorflow/core/distributed_runtime/integration_test:c_api_session_coordination_test_cpu \
-//tensorflow/core/grappler/optimizers:arithmetic_optimizer_test_cpu \
-//tensorflow/core/grappler/optimizers:auto_mixed_precision_test_cpu \
-//tensorflow/core/grappler/optimizers:constant_folding_test \
-//tensorflow/core/grappler/optimizers:remapper_test_cpu \
-//tensorflow/core/profiler/rpc/client:remote_profiler_session_manager_test \
-//tensorflow/python/kernel_tests/nn_ops:conv_ops_3d_test_cpu \
-//tensorflow/python/kernel_tests/nn_ops:conv2d_backprop_filter_grad_test_cpu \
-//tensorflow/python/tools:aot_compiled_test"
