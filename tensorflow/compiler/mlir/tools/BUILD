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
    name = "translate_cl_options",
    srcs = [
        "tf_mlir_translate_cl.cc",
    ],
    hdrs = [
        "tf_mlir_translate_cl.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translate_registration",
    srcs = [
        "tf_mlir_translate_registration.cc",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/translate:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow/translate/tools:file_tf_mlir_translate",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//tensorflow/compiler/mlir/tools:translate_cl_options",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/client:client_library",
        "@xla//xla/client:compile_only_client",
        "@xla//xla/service/cpu:cpu_compiler",
        "@xla//xla/service/cpu:cpu_transfer_manager",
        "@xla//xla/stream_executor/host:host_platform",
        "@xla//xla/stream_executor/host:host_platform_id",
    ],
    alwayslink = 1,
)

cc_library(
    name = "safe_cast",
    hdrs = ["safe_cast.h"],
)
