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

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_hardware",
    srcs = ["gpu_hardware.cc"],
    hdrs = ["gpu_hardware.h"],
    deps = [
        ":target_hardware",
        "//tensorflow/compiler/mlir/lite:cost_estimators",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:common",
        "//tensorflow/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cpu_hardware",
    srcs = ["cpu_hardware.cc"],
    deps = [
        ":target_hardware",
        "//tensorflow/compiler/mlir/lite:cost_estimators",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:common",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nnapi_hardware",
    srcs = ["nnapi_hardware.cc"],
    hdrs = ["nnapi_hardware.h"],
    deps = [
        ":target_hardware",
        "//tensorflow/compiler/mlir/lite:cost_estimators",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "//tensorflow/compiler/mlir/lite/experimental/tac/hardwares:simple_hardware",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "target_hardware",
    srcs = ["target_hardware.cc"],
    hdrs = ["target_hardware.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:common",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "simple_hardware",
    srcs = ["simple_hardware.cc"],
    hdrs = ["simple_hardware.h"],
    deps = [
        ":target_hardware",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "all-target-hardwares",
    deps = [
        ":cpu_hardware",
        ":gpu_hardware",
        ":nnapi_hardware",
    ],
    alwayslink = 1,
)
