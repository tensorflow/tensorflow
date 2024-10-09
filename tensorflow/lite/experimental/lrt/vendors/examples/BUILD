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

load("//tensorflow/lite/experimental/lrt/build_common:lite_rt_build_defs.bzl", "lite_rt_cc_lib_and_so")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

lite_rt_cc_lib_and_so(
    name = "mul_op_plugin",
    srcs = ["mul_op_plugin.cc"],
    shared_lib_name = "Plugin_Test_MulOp",
    visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
    deps = [
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
    ],
)

cc_test(
    name = "mul_op_plugin_test",
    srcs = ["mul_op_plugin_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":mul_op_plugin",  # buildcleaner: keep
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/test:common",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)
