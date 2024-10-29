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
    name = "litert_cc_api",
    srcs = [
        "litert_model.cc",
    ],
    hdrs = [
        "litert_any.h",
        "litert_handle.h",
        "litert_model.h",
        "litert_support.h",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
        "//tensorflow/lite/experimental/litert/core:model",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "any_cc_test",
    srcs = [
        "any_cc_test.cc",
    ],
    deps = [
        ":litert_cc_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "layout_cc_test",
    srcs = [
        "layout_cc_test.cc",
    ],
    deps = [
        ":litert_cc_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ranked_tensor_type_cc_test",
    srcs = [
        "ranked_tensor_type_cc_test.cc",
    ],
    deps = [
        ":litert_cc_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tensor_buffer",
    hdrs = [
        "litert_handle.h",
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tensor_buffer_test",
    srcs = [
        "tensor_buffer_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_buffer_requirements_test",
    srcs = [
        "tensor_buffer_requirements_test.cc",
    ],
    deps = [
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_cc_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "op_test",
    srcs = ["op_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_cc_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "subgraph_test",
    srcs = ["subgraph_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_cc_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_cc_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))
