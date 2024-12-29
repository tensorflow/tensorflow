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
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = [
        "model.h",
        "//tensorflow/lite/experimental/litert/c:litert_model_hdrs",
    ],
    deps = [
        ":ir_allocator",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_layout",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core:byte_code_util",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_load",
    srcs = ["model_load.cc"],
    hdrs = ["model_load.h"],
    deps = [
        ":flatbuffer_to_litert",
        ":model",
        ":model_graph",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "model_file_test",
    srcs = ["model_file_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":graph_validation",
        ":model",
        ":model_file_test_util",
        ":model_load",
        ":model_serialize",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_element_type",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core:byte_code_util",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "//tensorflow/lite/experimental/litert/test:test_models",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_serialize",
    srcs = ["model_serialize.cc"],
    hdrs = ["model_serialize.h"],
    deps = [
        ":litert_to_flatbuffer",
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/core:byte_code_util",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "flatbuffer_to_litert",
    srcs = ["flatbuffer_to_litert.cc"],
    hdrs = ["flatbuffer_to_litert.h"],
    deps = [
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_layout",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "flatbuffer_to_litert_test",
    srcs = ["flatbuffer_to_litert_test.cc"],
    deps = [
        ":flatbuffer_to_litert",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_to_flatbuffer",
    srcs = ["litert_to_flatbuffer.cc"],
    hdrs = ["litert_to_flatbuffer.h"],
    deps = [
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_to_flatbuffer_test",
    srcs = ["litert_to_flatbuffer_test.cc"],
    deps = [
        ":litert_to_flatbuffer",
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_layout",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_buffer",
    srcs = ["model_buffer.cc"],
    hdrs = ["model_buffer.h"],
    deps = [
        ":model",
        ":model_load",
        ":model_serialize",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/core:byte_code_util",
        "//tensorflow/lite/experimental/litert/core:filesystem",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "model_file_test_util",
    testonly = 1,
    srcs = ["model_file_test_util.cc"],
    hdrs = ["model_file_test_util.h"],
    deps = [
        ":flatbuffer_to_litert",
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ir_allocator",
    hdrs = ["ir_allocator.h"],
    deps = ["@com_google_absl//absl/types:span"],
)

cc_test(
    name = "ir_allocator_test",
    srcs = ["ir_allocator_test.cc"],
    deps = [
        ":ir_allocator",
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_graph",
    srcs = ["model_graph.cc"],
    hdrs = [
        "model_graph.h",
        "//tensorflow/lite/experimental/litert/cc:litert_consts.h",
    ],
    deps = [
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "graph_validation",
    srcs = ["graph_validation.cc"],
    hdrs = ["graph_validation.h"],
    deps = [
        ":model",
        ":model_graph",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
    ],
)

cc_test(
    name = "model_graph_test",
    srcs = ["model_graph_test.cc"],
    deps = [
        ":graph_validation",
        ":model",
        ":model_graph",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_buffer_test",
    srcs = ["model_buffer_test.cc"],
    deps = [
        ":model",
        ":model_buffer",
        ":model_load",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/experimental/litert/core:byte_code_util",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model_npu",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
