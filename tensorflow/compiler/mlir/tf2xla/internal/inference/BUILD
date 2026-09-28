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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

gentbl_cc_library(
    name = "inference_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"inference_passes.h.inc": [
        "-gen-pass-decls",
        "-name=TF2XLA",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "inference_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "inference_metrics_pass",
    srcs = ["inference_metrics_pass.cc"],
    hdrs = ["inference_passes.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":inference_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "inference_metrics_pass_test",
    srcs = ["inference_metrics_pass_test.cc"],
    deps = [
        "inference_metrics_pass",
        "//tensorflow/compiler/mlir/tf2xla/transforms:test_utils",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@xla//xla/tsl/platform:statusor",
    ],
)
