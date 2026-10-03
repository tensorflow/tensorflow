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
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# LINT.IfChange(cc_library_padding)
cc_library(
    name = "padding",
    srcs = [],
    hdrs = ["padding.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/compiler/mlir/lite/kernels:__pkg__"],
    deps = [
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
    ],
)
# LINT.ThenChange(//tensorflow/lite/kernels/BUILD)
