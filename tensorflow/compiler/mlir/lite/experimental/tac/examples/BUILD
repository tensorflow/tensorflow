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

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "example_hardware",
    srcs = ["example_hardware.cc"],
    hdrs = ["example_hardware.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "//tensorflow/compiler/mlir/lite/experimental/tac/hardwares:simple_hardware",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "example-hardware-translate",
    deps = [
        ":example_hardware",
        "//tensorflow/compiler/mlir/lite/experimental/tac:tac-translate-lib",
        "//tensorflow/compiler/mlir/lite/experimental/tac/hardwares:all-target-hardwares",
    ],
)
