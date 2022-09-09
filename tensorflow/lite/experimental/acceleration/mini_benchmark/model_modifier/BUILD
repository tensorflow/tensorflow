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
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "grafter",
    srcs = ["grafter.cc"],
    hdrs = ["grafter.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs_with_reflection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_library(
    name = "embedder",
    srcs = ["embedder.cc"],
    hdrs = ["embedder.h"],
    deps = [
        ":validation_graph_builder",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:decode_jpeg_status",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:jpeg_common",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:jpeg_header_parser",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_fbs_with_reflection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_library(
    name = "validation_graph_builder",
    srcs = ["validation_graph_builder.cc"],
    hdrs = ["validation_graph_builder.h"],
    deps = [
        ":grafter",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "embedder_cmdline",
    srcs = ["embedder_main.cc"],
    deps = [
        ":embedder",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:call",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:decode_jpeg",
        "//tensorflow/lite/schema:schema_fbs_with_reflection",
        "//tensorflow/lite/tools:command_line_flags",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_library(
    name = "custom_validation_embedder",
    srcs = ["custom_validation_embedder.cc"],
    hdrs = ["custom_validation_embedder.h"],
    deps = [
        ":constants",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

cc_test(
    name = "custom_validation_embedder_test",
    srcs = ["custom_validation_embedder_test.cc"],
    deps = [
        ":custom_validation_embedder",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:call",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:embedded_mobilenet_model",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:mini_benchmark_test_helper",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:model_loader",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)
