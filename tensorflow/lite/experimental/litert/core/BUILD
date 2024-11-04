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
    name = "api_internal",
    srcs = ["litert_common.cc"],
    hdrs = [
        "//tensorflow/lite/experimental/litert/c:litert_common.h",
        "//tensorflow/lite/experimental/litert/c:litert_logging.h",
        "//tensorflow/lite/experimental/litert/c:litert_model.h",
        "//tensorflow/lite/experimental/litert/c:litert_op_code.h",
        "//tensorflow/lite/experimental/litert/c:litert_options.h",
        "//tensorflow/lite/experimental/litert/c:litert_support.h",
        "//tensorflow/lite/experimental/litert/cc:litert_support.h",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin.h",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/core/c:c_api_types",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "model.h",
    ],
    deps = [
        ":api_internal",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/core/util:buffer_ref",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "option",
    srcs = ["option.cc"],
    hdrs = [
        "model.h",
    ],
    deps = [
        ":api_internal",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/core/util:buffer_ref",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_model_init",
    srcs = ["litert_model_init.cc"],
    hdrs = ["litert_model_init.h"],
    deps = [
        ":api_internal",
        ":model",
        ":option",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/core/util:buffer_ref",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "litert_model_serialize",
    srcs = ["litert_model_serialize.cc"],
    hdrs = ["litert_model_serialize.h"],
    deps = [
        ":litert_model_init",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "litert_model_serialize_test",
    srcs = ["litert_model_serialize_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":graph_tools",
        ":litert_model_init",
        ":litert_model_serialize",
        ":model",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "//tensorflow/lite/experimental/litert/core/util:buffer_ref",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":api_internal",
        ":graph_tools",
        ":litert_model_init",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/core/util:buffer_ref",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "option_test",
    srcs = ["option_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":api_internal",
        ":graph_tools",
        ":litert_model_init",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "graph_tools",
    hdrs = [
        "graph_tools.h",
    ],
    deps = [
        ":api_internal",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/core/util:buffer_ref",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "dynamic_loading",
    srcs = ["dynamic_loading.cc"],
    hdrs = ["dynamic_loading.h"],
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "dynamic_loading_test",
#     srcs = ["dynamic_loading_test.cc"],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#     ],
#     deps = [
#         ":dynamic_loading",
#         "@com_google_googletest//:gtest_main",
#         "//testing/base/public:unique-test-directory",
#         "@com_google_absl//absl/strings:string_view",
#         "//tensorflow/lite/experimental/litert/c:litert_c_api",
#         "//tensorflow/lite/experimental/litert/test:common",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "tensor_buffer",
    srcs = [
        "ahwb_buffer.cc",
        "dmabuf_buffer.cc",
        "event.cc",
        "fastrpc_buffer.cc",
        "ion_buffer.cc",
        "tensor_buffer.cc",
    ],
    hdrs = [
        "ahwb_buffer.h",
        "dmabuf_buffer.h",
        "event.h",
        "fastrpc_buffer.h",
        "ion_buffer.h",
        "tensor_buffer.h",
        "//tensorflow/lite/experimental/litert/c:litert_event.h",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer.h",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":utils",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tfl_utils",
    srcs = [
        "tfl_utils.cc",
    ],
    hdrs = [
        "tfl_utils.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
