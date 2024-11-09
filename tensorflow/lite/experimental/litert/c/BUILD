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
    name = "litert_common",
    hdrs = ["litert_common.h"],
)

cc_library(
    name = "litert_logging",
    srcs = [
        "litert_logging.cc",
    ],
    hdrs = [
        "litert_logging.h",
    ],
    deps = [
        ":litert_common",
        "//tensorflow/lite:minimal_logging",
    ],
)

cc_test(
    name = "litert_logging_test",
    srcs = [
        "litert_logging_test.cc",
    ],
    deps = [
        ":litert_common",
        ":litert_logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = ["litert_model.h"],
    deps = [
        ":litert_common",
        ":litert_op_code",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.cc"],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_op_code",
    hdrs = ["litert_op_code.h"],
    deps = ["//tensorflow/lite:builtin_ops"],
)

cc_library(
    name = "litert_options",
    srcs = ["litert_options.cc"],
    hdrs = [
        "litert_options.h",
    ],
    deps = [
        ":litert_common",
        ":litert_op_code",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/core/model",
    ],
)

cc_test(
    name = "litert_options_test",
    srcs = ["litert_options_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":litert_model",
        ":litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tensor_buffer",
    srcs = [
        "litert_event.cc",
        "litert_tensor_buffer.cc",
        "litert_tensor_buffer_requirements.cc",
    ],
    hdrs = [
        "litert_event.h",
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_common",
        ":litert_logging",
        ":litert_model",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_common",
        ":litert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "litert_dispatch_delegate.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/experimental/litert/runtime/dispatch:dispatch_delegate",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
    ],
)

filegroup(
    name = "litert_model_srcs",
    srcs = ["litert_model.cc"],
    visibility = ["//tensorflow/lite/experimental/litert/core/model:__pkg__"],
)

filegroup(
    name = "litert_model_hdrs",
    srcs = ["litert_model.h"],
    visibility = ["//tensorflow/lite/experimental/litert/core/model:__pkg__"],
)

exports_files(srcs = glob(["litert_*.h"]))
