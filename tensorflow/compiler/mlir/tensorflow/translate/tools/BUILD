# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "parsers",
    srcs = ["parsers.cc"],
    hdrs = ["parsers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@xla//xla:status_macros",
    ],
)

cc_library(
    name = "file_tf_mlir_translate",
    srcs = ["file_tf_mlir_translate.cc"],
    hdrs = ["file_tf_mlir_translate.h"],
    deps = [
        ":parsers",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/util/tensor_bundle:byteswaptensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)
