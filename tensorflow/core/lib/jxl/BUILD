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
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "hdrs",
    srcs = ["jxl_io.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "jxl_io",
    srcs = ["jxl_io.cc"],
    hdrs = ["jxl_io.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "@com_google_absl//absl/strings",
        "@jpegxl",
        "@jpegxl//:jpegxl_threads",
    ],
)

alias(
    name = "testdata",
    actual = "//tensorflow/core/lib/jxl/testdata:testdata",
    visibility = ["//tensorflow/core:__pkg__"],
)
