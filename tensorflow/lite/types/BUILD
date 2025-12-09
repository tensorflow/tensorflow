# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "fp16",
    hdrs = ["fp16.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "bit_cast",
    hdrs = ["bit_cast.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "half",
    hdrs = [
        "half.h",
    ],
    # copybara:uncomment_begin(google-only)
    # compatible_with = get_compatible_with_portable(),
    # copybara:uncomment_end
    deps = [
        ":bit_cast",
        ":fp16",
        "@FP16",
    ],
)
