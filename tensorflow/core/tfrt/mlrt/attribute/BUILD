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
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//smartass/brain/ops/tfrt_kernels:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/transforms/mlrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:__subpackages__",
    ],
)

cc_library(
    name = "attribute",
    srcs = ["attribute.cc"],
    hdrs = ["attribute.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "attribute_test",
    srcs = ["attribute_test.cc"],
    deps = [
        ":attribute",
        "//tensorflow/compiler/mlir/tensorflow",  # build_cleaner: keep
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@xla//xla/tsl/platform:status_matchers",
        "@xla//xla/tsl/platform:statusor",
    ],
)
