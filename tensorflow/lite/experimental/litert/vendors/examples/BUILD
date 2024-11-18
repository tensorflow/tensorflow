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

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "example_plugin",
    srcs = [
        "example_plugin.cc",
        "example_plugin_common.cc",
        "example_plugin_common.h",
    ],
    hdrs = ["//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "example_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Example.so",
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
    ],
)

litert_dynamic_lib(
    name = "example_plugin_with_convert_types",
    srcs = [
        "example_plugin_common.cc",
        "example_plugin_common.h",
        "example_plugin_with_convert_types.cc",
    ],
    hdrs = ["//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "example_plugin_with_convert_types_so",
    so_name = "libLiteRtCompilerPlugin_ExampleWithConvertTypes.so",
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":example_convert_types_impl",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_convert",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "example_plugin_test",
    srcs = [
        "example_plugin_test.cc",
    ],
    data = ["//tensorflow/lite/experimental/litert/test:tflite_test_data"],
    deps = [
        ":example_plugin",  # buildcleaner: keep
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "example_plugin_with_convert_types_test",
    srcs = ["example_plugin_with_convert_types_test.cc"],
    data = ["//tensorflow/lite/experimental/litert/test:tflite_test_data"],
    deps = [
        ":example_plugin_with_convert_types",  # buildcleaner: keep
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "example_convert_types_impl",
    hdrs = ["example_convert_types_impl.h"],
    visibility = ["//tensorflow/lite/experimental/litert/vendors/cc:__pkg__"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_convert",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_convert_types",
    ],
)
