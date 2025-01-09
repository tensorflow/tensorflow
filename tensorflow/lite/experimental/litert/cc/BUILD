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
    default_visibility = [
        # copybara:uncomment "//third_party/mediapipe/calculators/tensor:__subpackages__",
        # copybara:uncomment "//third_party/odml/infra:__subpackages__",
        "//tensorflow/lite/experimental/litert:__subpackages__",
    ],
)

cc_library(
    name = "litert_environment",
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_any",
        ":litert_expected",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_event",
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_expected",
        ":litert_handle",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_event",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
    deps = [
        ":litert_expected",
        "//tensorflow/lite/experimental/litert/c:litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_any_test",
    srcs = [
        "litert_any_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = [
        "litert_consts.h",
        "litert_model.h",
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_detail",
        ":litert_element_type",
        ":litert_expected",
        ":litert_handle",
        ":litert_layout",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = [
        "litert_model_test.cc",
    ],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_layout",
        ":litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_handle",
    hdrs = ["litert_handle.h"],
)

cc_library(
    name = "litert_tensor_buffer",
    hdrs = [
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_event",
        ":litert_handle",
        ":litert_model",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_event",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
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
        ":litert_layout",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tensor_buffer_requirements",
    hdrs = [
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_handle",
        ":litert_macros",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    deps = [
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_buffer_ref",
    hdrs = [
        "litert_buffer_ref.h",
    ],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_macros",
    hdrs = ["litert_macros.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "litert_expected",
    hdrs = ["litert_expected.h"],
    deps = [
        ":litert_detail",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_expected_test",
    srcs = ["litert_expected_test.cc"],
    deps = [
        ":litert_buffer_ref",
        ":litert_expected",
        ":litert_macros",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_detail",
    hdrs = ["litert_detail.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_test(
    name = "litert_buffer_ref_test",
    srcs = ["litert_buffer_ref_test.cc"],
    deps = [
        ":litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_element_type",
    hdrs = ["litert_element_type.h"],
    deps = ["//tensorflow/lite/experimental/litert/c:litert_model"],
)

cc_test(
    name = "litert_element_type_test",
    srcs = ["litert_element_type_test.cc"],
    deps = [
        ":litert_element_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_model_predicates",
    srcs = ["litert_model_predicates.cc"],
    hdrs = ["litert_model_predicates.h"],
    deps = [
        ":litert_detail",
        ":litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_layout",
    hdrs = ["litert_layout.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_layout",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_predicates_test",
    srcs = ["litert_model_predicates_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_model",
        ":litert_model_predicates",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_layout_test",
    srcs = ["litert_layout_test.cc"],
    deps = [
        ":litert_layout",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["litert_compiled_model.cc"],
    hdrs = ["litert_compiled_model.h"],
    deps = [
        ":litert_detail",
        ":litert_expected",
        ":litert_handle",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model_options",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = ["litert_compiled_model_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_compiled_model",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))
