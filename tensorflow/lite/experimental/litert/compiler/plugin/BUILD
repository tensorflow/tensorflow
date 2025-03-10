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
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["compiler_plugin.h"],
    deps = [
        ":algo",
        ":compiler_flags",
        "//tensorflow/lite/experimental/litert/c:litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_op_options",
        "//tensorflow/lite/experimental/litert/cc:litert_shared_library",
        "//tensorflow/lite/experimental/litert/core:build_stamp",
        "//tensorflow/lite/experimental/litert/core:dynamic_loading",
        "//tensorflow/lite/experimental/litert/core:environment",
        "//tensorflow/lite/experimental/litert/core:filesystem",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/model:buffer_manager",
        "//tensorflow/lite/experimental/litert/core/model:ir_allocator",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin_api",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "compiler_plugin_test",
#     srcs = ["compiler_plugin_test.cc"],
#     data = [
#         "//tensorflow/lite/experimental/litert/test:mlir_test_data",
#         "//tensorflow/lite/experimental/litert/vendors/examples:example_plugin_so",
#     ],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#         "notsan",
#     ],
#     deps = [
#         ":compiler_plugin",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/strings:string_view",
#         "//tensorflow/lite/experimental/litert/c:litert_common",
#         "//tensorflow/lite/experimental/litert/c:litert_model",
#         "//tensorflow/lite/experimental/litert/c:litert_op_code",
#         "//tensorflow/lite/experimental/litert/cc:litert_environment",
#         "//tensorflow/lite/experimental/litert/cc:litert_op_options",
#         "//tensorflow/lite/experimental/litert/core:build_stamp",
#         "//tensorflow/lite/experimental/litert/core:filesystem",
#         "//tensorflow/lite/experimental/litert/core/model",
#         "//tensorflow/lite/experimental/litert/test:common",
#         "//tensorflow/lite/experimental/litert/test:matchers",
#         "//tensorflow/lite/experimental/litert/tools:dump",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "algo",
    srcs = ["algo.cc"],
    hdrs = ["algo.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/core:insert_order_map",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/model:model_graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_test(
    name = "algo_test",
    srcs = ["algo_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
    ],
    deps = [
        ":algo",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/model:graph_validation",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "compiler_flags",
    srcs = ["compiler_flags.cc"],
    hdrs = ["compiler_flags.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "compiler_flags_test",
    srcs = ["compiler_flags_test.cc"],
    deps = [
        ":compiler_flags",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
