# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "mul_op_plugin",
    srcs = ["mul_op_plugin.cc"],
    deps = [
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:graph_tools",
    ],
)

cc_shared_library(
    name = "mul_op_plugin_so",
    shared_lib_name = "mul_op_plugin.so",
    visibility = ["//tensorflow/compiler/mlir/lite/experimental/lrt:__subpackages__"],
    deps = [":mul_op_plugin"],
)

cc_test(
    name = "mul_op_plugin_test",
    srcs = ["mul_op_plugin_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":mul_op_plugin",  # buildcleaner: keep
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:model",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/test_data:test_data_util",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)
