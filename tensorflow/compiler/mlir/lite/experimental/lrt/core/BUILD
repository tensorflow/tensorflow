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
    default_visibility = ["//tensorflow/compiler/mlir/lite/experimental/lrt:__subpackages__"],
)

cc_library(
    name = "common_internal",
    srcs = ["lite_rt_common.cc"],
    deps = ["//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api"],
)

cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = [
        "model.h",
    ],
    deps = [
        ":common_internal",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "lite_rt_model_init",
    srcs = ["lite_rt_model_init.cc"],
    hdrs = ["lite_rt_model_init.h"],
    deps = [
        ":common_internal",
        ":model",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":common_internal",
        ":graph_tools",
        ":lite_rt_model_init",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/test_data:test_data_util",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "algo",
    hdrs = ["algo.h"],
    deps = [
        ":common_internal",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:model",
    ],
)

cc_test(
    name = "algo_test",
    srcs = ["algo_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":algo",
        ":common_internal",
        ":graph_tools",
        ":model",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/test_data:test_data_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "graph_tools",
    hdrs = [
        "graph_tools.h",
    ],
    deps = [
        ":common_internal",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/c:c_api_types",
        "@llvm-project//llvm:Support",
    ],
)
