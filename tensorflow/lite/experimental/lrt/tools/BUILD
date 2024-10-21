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
    default_visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
)

cc_library(
    name = "apply_plugin",
    testonly = 1,
    srcs = ["apply_plugin.cc"],
    hdrs = ["apply_plugin.h"],
    deps = [
        ":dump",
        ":tool_display",
        "//tensorflow/lite/experimental/lrt/c:litert_c_api",
        "//tensorflow/lite/experimental/lrt/cc:litert_cc_api",
        "//tensorflow/lite/experimental/lrt/core:litert_model_init",
        "//tensorflow/lite/experimental/lrt/core/compiler_plugin",
        "//tensorflow/lite/experimental/lrt/core/compiler_plugin:algo",
        "//tensorflow/lite/experimental/lrt/core/util:buffer_ref",
        "//tensorflow/lite/experimental/lrt/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/lrt/test:common",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "apply_plugin_test",
    srcs = ["apply_plugin_test.cc"],
    data = [
        "//tensorflow/lite/experimental/lrt/test:tflite_test_data",
        "//tensorflow/lite/experimental/lrt/vendors/examples:example_plugin_so",
    ],
    tags = [
        "noasan",
        "nomsan",
        "nosan",
    ],
    deps = [
        ":apply_plugin",
        "//tensorflow/lite/experimental/lrt/c:litert_c_api",
        "//tensorflow/lite/experimental/lrt/core:litert_model_init",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/test:common",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "apply_plugin_main",
    testonly = 1,
    srcs = ["apply_plugin_main.cc"],
    data = ["//tensorflow/lite/experimental/lrt/vendors/examples:example_plugin_so"],
    linkstatic = 1,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
    ],
    deps = [
        ":apply_plugin",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tool_display",
    srcs = ["tool_display.cc"],
    hdrs = ["tool_display.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "tool_display_test",
    srcs = ["tool_display_test.cc"],
    data = ["//tensorflow/lite/experimental/lrt/test:tflite_test_data"],
    deps = [
        ":tool_display",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    deps = [
        "//tensorflow/lite/experimental/lrt/c:litert_c_api",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/core/compiler_plugin",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "dump_test",
    srcs = ["dump_test.cc"],
    data = ["//tensorflow/lite/experimental/lrt/test:tflite_test_data"],
    deps = [
        ":dump",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/test:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
