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

# load(
#     "//tensorflow/core/platform:rules_cc.bzl",
#     "cc_library",
# )

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

# cc_library(
#     name = "bfloat16",
#     hdrs = ["bfloat16.h"],
#     deps = [
#         "@eigen_archive//:eigen3",
#         "//tensorflow/core/platform:byte_order",
#     ],
# )

# # Export source files needed for mobile builds, which do not use granular targets.
# filegroup(
#     name = "mobile_srcs_no_runtime",
#     srcs = [
#         "bfloat16.h",
#     ],
# )

exports_files(
    ["bfloat16.h"],
)
