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
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin",
    ],
)

cc_library(
    name = "conversion",
    hdrs = ["conversion.h"],
    deps = [
        ":backend_ir",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "backend_ir",
    hdrs = ["backend_ir.h"],
    deps = ["//tensorflow/lite/experimental/litert/c:litert_common"],
)

cc_library(
    name = "partition_with_capabilities",
    hdrs = ["partition_with_capabilities.h"],
    deps = [
        ":backend_ir",
        ":conversion",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
    ],
)

cc_test(
    name = "partition_with_capabilities_test",
    srcs = ["partition_with_capabilities_test.cc"],
    deps = [
        ":conversion",
        ":partition_with_capabilities",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/model:model_graph",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/vendors/examples:example_conversion_impl",
        "//tensorflow/lite/experimental/litert/vendors/examples:example_ir",
        "@com_google_googletest//:gtest_main",
    ],
)
