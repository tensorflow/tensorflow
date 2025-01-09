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
    name = "event",
    srcs = [
        "event.cc",
    ],
    hdrs = [
        "event.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
    ],
)

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
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_event",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_event",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core/util:tensor_type_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
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
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
        "//tensorflow/lite/experimental/litert/cc:litert_element_type",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
    ],
)

cc_library(
    name = "external_litert_buffer_context",
    srcs = ["external_litert_buffer_context.cc"],
    hdrs = ["external_litert_buffer_context.h"],
    deps = [
        ":tfl_utils",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer_requirements",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "compiled_model",
    srcs = ["compiled_model.cc"],
    hdrs = ["compiled_model.h"],
    deps = [
        ":external_litert_buffer_context",
        ":tensor_buffer",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model_options",
        "//tensorflow/lite/experimental/litert/c:litert_dispatch_delegate",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
        "//tensorflow/lite/experimental/litert/cc:litert_event",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer_requirements",
        "//tensorflow/lite/experimental/litert/compiler/plugin:compiler_plugin",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/model:model_serialize",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "compiled_model_test",
    srcs = ["compiled_model_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":compiled_model",
        ":tensor_buffer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model_options",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_environment",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
