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

load("//tensorflow:tensorflow.default.bzl", "filegroup")

# Library for using LLVM style RTTI in TensorFlow.
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "llvm_rtti",
    hdrs = ["llvm_rtti.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "legacy_lib_core_all_headers",
    srcs = [
        "llvm_rtti.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
