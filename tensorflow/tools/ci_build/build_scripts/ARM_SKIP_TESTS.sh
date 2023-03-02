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

ARM_SKIP_TESTS="-//tensorflow/lite/... \
-//tensorflow/core/platform:ram_file_system_test \
-//tensorflow/python/client:session_list_devices_test \
-//tensorflow/python/compiler/xla:xla_test_cpu \
-//tensorflow/python/compiler/xla:xla_test_gpu \
-//tensorflow/python/data/experimental/kernel_tests:checkpoint_input_pipeline_hook_test \
-//tensorflow/python/data/kernel_tests:iterator_test \
-//tensorflow/python/distribute:parameter_server_strategy_test_cpu \
-//tensorflow/python/distribute:parameter_server_strategy_test_gpu \
-//tensorflow/python/distribute/failure_handling:gce_failure_handler_test \
-//tensorflow/python/kernel_tests/nn_ops:atrous_conv2d_test \
-//tensorflow/python/training:server_lib_test \
-//tensorflow/python/debug/lib:source_remote_test"
